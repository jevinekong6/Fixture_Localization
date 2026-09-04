"""Landmark registry: spatial gating of observations against an apriori map.

ROS-free by design, so the association policy can be tested exhaustively
without a robot, a camera, or a running graph.

THE GOVERNING RULE
------------------
CAD POSITIONS ARE NEVER OVERWRITTEN BY OBSERVATION. Disagreement produces a
FLAG, not a silent edit.

An apriori landmark came from a drawing that someone is accountable for. A
depth-from-known-size observation came from a bounding box scaled by a guessed
real-world width, at an error that grows with the square of range. When the two
disagree, the observation is overwhelmingly the more likely to be wrong, and
quietly averaging it into the CAD entry would destroy the one number in the
system that is independently trustworthy -- and destroy it invisibly, leaving a
map that looks converged and is subtly wrong everywhere. So a disagreement gets
recorded as a residual and surfaced as a red marker, and a human decides
whether the drawing or the estimator is at fault.

Observed-only landmarks have no such provenance, so those DO get refined by
inverse-variance weighting as more observations arrive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

__all__ = [
    "Origin",
    "Outcome",
    "Landmark",
    "Association",
    "FixtureRegistry",
]


class Origin(Enum):
    """Where a landmark's position came from."""

    APRIORI = "apriori"    # from CAD / a survey. Authoritative. Never moved.
    OBSERVED = "observed"  # built from observations alone. Refinable.


class Outcome(Enum):
    """What happened to an observation."""

    CONFIRMED = "confirmed"  # matched a CAD entry, close enough
    REFINED = "refined"      # matched an observed-only landmark, position updated
    FLAGGED = "flagged"      # matched a CAD entry, but too far off -- needs a human
    NEW = "new"              # matched nothing, created an observed-only landmark
    REJECTED = "rejected"    # out of trusted range, discarded before association


@dataclass
class Landmark:
    """One fixture in the map."""

    id: str
    cls: str
    position: np.ndarray               # 3-vector in the map frame, metres
    origin: Origin
    sigma: float = 0.5                 # isotropic 1-sigma position uncertainty, metres
    n_obs: int = 0                     # how many observations have hit it
    patches: List[str] = field(default_factory=list)  # saved crop filenames
    max_residual: float = 0.0          # worst observation-to-CAD distance seen

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)

    def as_dict(self) -> Dict:
        """Plain-Python view for YAML. No numpy scalars, no enums."""
        return {
            "id": str(self.id),
            "class": str(self.cls),
            "position": [float(v) for v in self.position],
            "origin": self.origin.value,
            "sigma": float(self.sigma),
            "n_obs": int(self.n_obs),
            "patches": [str(p) for p in self.patches],
            "max_residual": float(self.max_residual),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Landmark":
        return cls(
            id=str(d["id"]),
            cls=str(d.get("class", d.get("cls", ""))),
            position=np.asarray(d["position"], dtype=float).reshape(3),
            origin=Origin(str(d.get("origin", "apriori"))),
            sigma=float(d.get("sigma", 0.5)),
            n_obs=int(d.get("n_obs", 0)),
            patches=list(d.get("patches") or []),
            max_residual=float(d.get("max_residual", 0.0)),
        )


@dataclass
class Association:
    """The result of feeding one observation to the registry."""

    outcome: Outcome
    landmark: Optional[Landmark]  # None only for REJECTED
    residual: Optional[float]     # distance to the matched landmark, metres
    note: str


class FixtureRegistry:
    """Nearest-neighbour, within-class association against an apriori map.

    Radii, all in metres:
        confirm_radius  an observation this close to a CAD entry confirms it
        flag_radius     ... this close, but not that close, flags it
        assoc_radius    gate for merging into an existing observed-only landmark
        max_range       observations beyond this range are discarded outright

    ASSOCIATION IS NEAREST-NEIGHBOUR WITHIN CLASS, which is only valid while
    each class has at most ONE INSTANCE IN VIEW. With two knobs a metre apart,
    a range error of half a metre -- entirely ordinary at 4 m -- will attach the
    observation to the wrong knob, and the registry has no way to notice. Two
    real fixes exist, neither implemented here because neither is meaningful at
    the tripod stage: (a) project-and-gate, where a pose prior predicts each
    known landmark's image location and association happens in pixels where the
    bearing is accurate, rather than in metres where the range is not; or
    (b) geometric-consistency RANSAC over the whole detection set, accepting the
    assignment whose inter-landmark distances match the map. Until one of those
    exists, put one fixture per class in the scene.
    """

    def __init__(
        self,
        confirm_radius: float = 0.15,
        flag_radius: float = 0.60,
        assoc_radius: float = 0.30,
        max_range: float = 6.0,
    ) -> None:
        if not 0.0 < confirm_radius <= flag_radius:
            raise ValueError("need 0 < confirm_radius <= flag_radius")
        self.confirm_radius = float(confirm_radius)
        self.flag_radius = float(flag_radius)
        self.assoc_radius = float(assoc_radius)
        self.max_range = float(max_range)
        self.landmarks: List[Landmark] = []
        self.frame_id: str = "map"
        self._obs_counter: Dict[str, int] = {}

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #
    def add_landmark(self, landmark: Landmark) -> Landmark:
        """Insert a landmark verbatim. Used by load_apriori and by tests."""
        if any(lm.id == landmark.id for lm in self.landmarks):
            raise ValueError(f"duplicate landmark id {landmark.id!r}")
        self.landmarks.append(landmark)
        return landmark

    def load_apriori(self, path) -> int:
        """Load a map YAML (``frame_id`` plus a ``landmarks`` list).

        Entries keep whatever ``origin`` the file gives them, so the output map
        written by a previous run can be fed straight back in. Returns the
        number of landmarks loaded.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"apriori map not found: {path}")
        with open(path, "r") as f:
            doc = yaml.safe_load(f) or {}

        self.frame_id = str(doc.get("frame_id", "map"))
        for entry in doc.get("landmarks") or []:
            lm = Landmark.from_dict(entry)
            self.add_landmark(lm)
            if lm.origin is Origin.OBSERVED:
                # Keep generated ids monotonic across a reload.
                n = _trailing_index(lm.id)
                if n is not None:
                    self._obs_counter[lm.cls] = max(self._obs_counter.get(lm.cls, 0), n)
        return len(self.landmarks)

    def save(self, path) -> None:
        """Write the whole registry -- apriori and observed alike -- to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "frame_id": self.frame_id,
            "landmarks": [lm.as_dict() for lm in self.landmarks],
        }
        with open(path, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)

    # ----------------------------------------------------------------- #
    # The association policy
    # ----------------------------------------------------------------- #
    def observe(
        self,
        cls: str,
        position,
        sigma: float,
        range_m: Optional[float] = None,
        patch: Optional[str] = None,
    ) -> Association:
        """Feed one map-frame observation to the registry.

        Order of the gates, which matters:

        1. Beyond ``max_range`` -> REJECTED, and nothing is created or touched.
           This runs FIRST: a 9 m observation carries a metre of range error, so
           letting it vote on association at all is worse than dropping it.
        2. Nearest APRIORI landmark of the same class within ``confirm_radius``
           -> CONFIRMED. n_obs increments. The CAD position does not move.
        3. Between ``confirm_radius`` and ``flag_radius`` -> FLAGGED, residual
           recorded. The CAD position still does not move.
        4. Otherwise nearest OBSERVED landmark within ``assoc_radius``
           -> REFINED by an inverse-variance weighted update.
        5. Otherwise -> NEW observed-only landmark.
        """
        position = np.asarray(position, dtype=float).reshape(3)
        sigma = float(sigma)
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # 1. Range gate, before anything else.
        if range_m is not None and range_m > self.max_range:
            return Association(
                outcome=Outcome.REJECTED,
                landmark=None,
                residual=None,
                note=(f"range {range_m:.2f} m exceeds max_range "
                      f"{self.max_range:.2f} m -- discarded before association"),
            )

        # 2 & 3. Apriori gate.
        cad, d_cad = self._nearest(cls, position, Origin.APRIORI)
        if cad is not None and d_cad <= self.confirm_radius:
            cad.n_obs += 1
            if patch:
                cad.patches.append(patch)
            return Association(
                outcome=Outcome.CONFIRMED,
                landmark=cad,
                residual=d_cad,
                note=f"within {self.confirm_radius:.2f} m of CAD ({d_cad:.3f} m)",
            )

        if cad is not None and d_cad <= self.flag_radius:
            cad.n_obs += 1
            cad.max_residual = max(cad.max_residual, d_cad)
            if patch:
                cad.patches.append(patch)
            return Association(
                outcome=Outcome.FLAGGED,
                landmark=cad,
                residual=d_cad,
                note=(f"observation is {d_cad:.3f} m from CAD entry {cad.id!r} "
                      f"(confirm radius {self.confirm_radius:.2f} m) -- CAD position "
                      f"left untouched, check width_m and the tripod pose"),
            )

        # 4. Observed-only refinement.
        obs, d_obs = self._nearest(cls, position, Origin.OBSERVED)
        if obs is not None and d_obs <= self.assoc_radius:
            w_old = 1.0 / (obs.sigma ** 2)
            w_new = 1.0 / (sigma ** 2)
            obs.position = (obs.position * w_old + position * w_new) / (w_old + w_new)
            obs.sigma = math.sqrt(1.0 / (w_old + w_new))
            obs.n_obs += 1
            if patch:
                obs.patches.append(patch)
            return Association(
                outcome=Outcome.REFINED,
                landmark=obs,
                residual=d_obs,
                note=f"merged into {obs.id!r}, sigma now {obs.sigma:.3f} m",
            )

        # 5. Brand new.
        n = self._obs_counter.get(cls, 0) + 1
        self._obs_counter[cls] = n
        lm = Landmark(
            id=f"{cls}_obs_{n:03d}",
            cls=cls,
            position=position.copy(),
            origin=Origin.OBSERVED,
            sigma=sigma,
            n_obs=1,
            patches=[patch] if patch else [],
        )
        self.add_landmark(lm)
        return Association(
            outcome=Outcome.NEW,
            landmark=lm,
            residual=None,
            note=f"no {cls} landmark within gate -- created {lm.id!r}",
        )

    # ----------------------------------------------------------------- #
    # Queries
    # ----------------------------------------------------------------- #
    def correspondences(self, confirmed_only: bool = True) -> List[Tuple[str, np.ndarray]]:
        """PnP-ready 3D points as ``[(landmark_id, position), ...]``.

        Defaults to CAD-origin landmarks only, and among those only ones that
        have actually been seen and were not flagged. An observed-only position
        is nothing but the estimator's own output: feeding it to PnP as a known
        3D point launders the depth-from-known-size error into a pose solution
        that then looks confident. A flagged CAD entry is excluded for the
        mirror-image reason -- something about it is unresolved, so it is not
        yet a correspondence anyone should be solving against.
        """
        out: List[Tuple[str, np.ndarray]] = []
        for lm in self.landmarks:
            if lm.n_obs == 0:
                continue
            if confirmed_only:
                if lm.origin is not Origin.APRIORI:
                    continue
                if lm.max_residual > self.confirm_radius:
                    continue
            out.append((lm.id, lm.position.copy()))
        return out

    def flagged(self) -> List[Landmark]:
        """CAD landmarks whose observations disagreed with the drawing."""
        return [lm for lm in self.landmarks
                if lm.origin is Origin.APRIORI and lm.max_residual > self.confirm_radius]

    def _nearest(self, cls: str, position: np.ndarray, origin: Origin):
        """Nearest landmark of the given class AND origin, with its distance."""
        best, best_d = None, float("inf")
        for lm in self.landmarks:
            if lm.cls != cls or lm.origin is not origin:
                continue
            d = float(np.linalg.norm(lm.position - position))
            if d < best_d:
                best, best_d = lm, d
        return best, best_d


def _trailing_index(landmark_id: str) -> Optional[int]:
    """Pull the ``007`` out of ``knob_obs_007``. None if it isn't there."""
    tail = landmark_id.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return None
