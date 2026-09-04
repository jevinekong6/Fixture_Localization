"""Unit tests for the ROS-free core of fixture_map_test.

These import nothing from ROS. Run them with:

    python3 -m pytest test/ -q

The first few tests are frame-convention canaries. If they fail, every
downstream map position is wrong in a way that still looks plausible on
screen, so treat them as the gate for everything else in this package.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixture_map_test.fixture_geometry import (  # noqa: E402
    BBox,
    Intrinsics,
    backproject,
    estimate_range,
    observe,
    obliquity_proxy,
)
from fixture_map_test.fixture_registry import (  # noqa: E402
    FixtureRegistry,
    Landmark,
    Origin,
    Outcome,
)

# A plausible ZED-ish left-camera intrinsic set, 1280x720.
INTR = Intrinsics(fx=700.0, fy=700.0, cx=640.0, cy=360.0)


# --------------------------------------------------------------------------- #
# Geometry: frame convention
# --------------------------------------------------------------------------- #
def test_on_axis_backprojects_to_optical_axis():
    """THE canary. A fixture dead centre in the image at range d must land at
    exactly (0, 0, d) in the optical frame -- no x, no y, all z."""
    d = 2.5
    p = backproject(BBox(cx=INTR.cx, cy=INTR.cy, w=50.0, h=50.0), INTR, d)
    assert p.shape == (3,)
    np.testing.assert_allclose(p, np.array([0.0, 0.0, d]), atol=1e-12)


def test_fixture_right_of_centre_gives_positive_x():
    p = backproject(BBox(cx=INTR.cx + 100.0, cy=INTR.cy, w=50.0, h=50.0), INTR, 2.0)
    assert p[0] > 0.0
    assert p[1] == pytest.approx(0.0, abs=1e-12)
    assert p[0] == pytest.approx(100.0 * 2.0 / INTR.fx)


def test_fixture_below_centre_gives_positive_y():
    """Optical +y is DOWN. A fixture lower in the image (larger v) is +y."""
    p = backproject(BBox(cx=INTR.cx, cy=INTR.cy + 80.0, w=50.0, h=50.0), INTR, 2.0)
    assert p[1] > 0.0
    assert p[0] == pytest.approx(0.0, abs=1e-12)
    assert p[1] == pytest.approx(80.0 * 2.0 / INTR.fy)


def test_fixture_above_centre_gives_negative_y():
    p = backproject(BBox(cx=INTR.cx, cy=INTR.cy - 80.0, w=50.0, h=50.0), INTR, 2.0)
    assert p[1] < 0.0


def test_pixel_to_point_to_pixel_roundtrip():
    """Back-project then reproject with the pinhole model and land on the
    original pixel. Catches an fx/fy or cx/cy swap."""
    u, v, z = 812.0, 275.0, 3.4
    p = backproject(BBox(cx=u, cy=v, w=40.0, h=40.0), INTR, z)
    u_re = INTR.fx * p[0] / p[2] + INTR.cx
    v_re = INTR.fy * p[1] / p[2] + INTR.cy
    assert u_re == pytest.approx(u, abs=1e-9)
    assert v_re == pytest.approx(v, abs=1e-9)


# --------------------------------------------------------------------------- #
# Geometry: range and uncertainty
# --------------------------------------------------------------------------- #
def test_range_is_inversely_proportional_to_bbox_width():
    w_real = 0.09
    near = estimate_range(BBox(640.0, 360.0, 100.0, 100.0), INTR, real_width_m=w_real)
    far = estimate_range(BBox(640.0, 360.0, 50.0, 50.0), INTR, real_width_m=w_real)
    assert near.z == pytest.approx(INTR.fx * w_real / 100.0)
    assert far.z == pytest.approx(2.0 * near.z)
    assert near.used_dim == "width"


def test_sigma_scales_with_range_squared():
    """4x the range (1/4 the bbox width) must give 16x the sigma."""
    w_real = 0.09
    near = estimate_range(BBox(640.0, 360.0, 200.0, 200.0), INTR,
                          real_width_m=w_real, bbox_sigma_px=3.0)
    far = estimate_range(BBox(640.0, 360.0, 50.0, 50.0), INTR,
                         real_width_m=w_real, bbox_sigma_px=3.0)
    assert far.z == pytest.approx(4.0 * near.z)
    assert far.sigma_z == pytest.approx(16.0 * near.sigma_z, rel=1e-9)


def test_sigma_matches_the_analytic_derivative():
    w_real, sigma_px = 0.09, 3.0
    est = estimate_range(BBox(640.0, 360.0, 120.0, 120.0), INTR,
                         real_width_m=w_real, bbox_sigma_px=sigma_px)
    expected = (est.z ** 2 / (INTR.fx * w_real)) * sigma_px
    assert est.sigma_z == pytest.approx(expected)


def test_prefer_height_uses_fy_and_bbox_height():
    est = estimate_range(BBox(640.0, 360.0, 100.0, 200.0), INTR,
                         real_width_m=0.09, real_height_m=0.09, prefer="height")
    assert est.used_dim == "height"
    assert est.z == pytest.approx(INTR.fy * 0.09 / 200.0)


def test_falls_back_to_the_dimension_that_was_supplied():
    est = estimate_range(BBox(640.0, 360.0, 100.0, 200.0), INTR,
                         real_height_m=0.09, prefer="width")
    assert est.used_dim == "height"


def test_zero_width_bbox_raises():
    with pytest.raises(ValueError):
        estimate_range(BBox(640.0, 360.0, 0.0, 50.0), INTR, real_width_m=0.09)


def test_negative_bbox_dimension_raises():
    with pytest.raises(ValueError):
        estimate_range(BBox(640.0, 360.0, -10.0, 50.0), INTR, real_width_m=0.09)


def test_no_real_size_raises():
    with pytest.raises(ValueError):
        estimate_range(BBox(640.0, 360.0, 50.0, 50.0), INTR)


def test_intrinsics_from_camera_info_k():
    k = [700.0, 0.0, 640.0, 0.0, 705.0, 360.0, 0.0, 0.0, 1.0]
    intr = Intrinsics.from_k(k)
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (700.0, 705.0, 640.0, 360.0)


def test_observe_returns_point_and_estimate_consistently():
    bbox = BBox(740.0, 300.0, 90.0, 90.0)
    point, est = observe(bbox, INTR, real_width_m=0.09)
    assert point[2] == pytest.approx(est.z)
    np.testing.assert_allclose(point, backproject(bbox, INTR, est.z))


# --------------------------------------------------------------------------- #
# Geometry: obliquity proxy
# --------------------------------------------------------------------------- #
def test_obliquity_proxy_is_one_for_fronto_parallel():
    assert obliquity_proxy(BBox(0.0, 0.0, 100.0, 100.0), nominal_aspect=1.0) == pytest.approx(1.0)


def test_obliquity_proxy_flags_a_foreshortened_box():
    """A wheel seen at a slant is squashed horizontally: aspect drops below 1."""
    ratio = obliquity_proxy(BBox(0.0, 0.0, 60.0, 100.0), nominal_aspect=1.0)
    assert ratio == pytest.approx(0.6)
    assert abs(ratio - 1.0) > 0.25


def test_obliquity_proxy_rejects_bad_nominal():
    with pytest.raises(ValueError):
        obliquity_proxy(BBox(0.0, 0.0, 60.0, 100.0), nominal_aspect=0.0)


# --------------------------------------------------------------------------- #
# Registry: apriori (CAD) behaviour
# --------------------------------------------------------------------------- #
CAD_POS = np.array([2.0, 0.5, 1.0])


def _registry_with_cad(**kw):
    reg = FixtureRegistry(**kw)
    reg.add_landmark(Landmark(
        id="valve_wheel_cad_01", cls="valve_wheel", position=CAD_POS.copy(),
        origin=Origin.APRIORI, sigma=0.01,
    ))
    return reg


def test_observation_near_cad_confirms_and_does_not_move_it():
    reg = _registry_with_cad()
    assoc = reg.observe("valve_wheel", CAD_POS + np.array([0.05, 0.0, 0.0]), sigma=0.2)
    assert assoc.outcome is Outcome.CONFIRMED
    assert assoc.landmark.id == "valve_wheel_cad_01"
    assert assoc.landmark.n_obs == 1
    np.testing.assert_allclose(assoc.landmark.position, CAD_POS)
    assert len(reg.landmarks) == 1


def test_observation_disagreeing_with_cad_flags_and_still_does_not_move_it():
    reg = _registry_with_cad()
    assoc = reg.observe("valve_wheel", CAD_POS + np.array([0.0, 0.35, 0.0]), sigma=0.2)
    assert assoc.outcome is Outcome.FLAGGED
    np.testing.assert_allclose(assoc.landmark.position, CAD_POS)
    assert assoc.residual == pytest.approx(0.35)
    assert "0.35" in assoc.note
    assert assoc.landmark.max_residual == pytest.approx(0.35)
    assert len(reg.landmarks) == 1


def test_max_residual_keeps_the_worst_not_the_latest():
    reg = _registry_with_cad()
    reg.observe("valve_wheel", CAD_POS + np.array([0.0, 0.50, 0.0]), sigma=0.2)
    reg.observe("valve_wheel", CAD_POS + np.array([0.0, 0.25, 0.0]), sigma=0.2)
    assert reg.landmarks[0].max_residual == pytest.approx(0.50)


def test_far_observation_of_same_class_creates_a_new_landmark():
    reg = _registry_with_cad()
    assoc = reg.observe("valve_wheel", CAD_POS + np.array([3.0, 0.0, 0.0]), sigma=0.2)
    assert assoc.outcome is Outcome.NEW
    assert assoc.landmark.origin is Origin.OBSERVED
    assert assoc.landmark.id == "valve_wheel_obs_001"
    assert len(reg.landmarks) == 2
    np.testing.assert_allclose(reg.landmarks[0].position, CAD_POS)


def test_different_class_never_associates_to_a_cad_entry():
    reg = _registry_with_cad()
    assoc = reg.observe("knob", CAD_POS.copy(), sigma=0.2)
    assert assoc.outcome is Outcome.NEW
    assert assoc.landmark.cls == "knob"
    assert reg.landmarks[0].n_obs == 0


# --------------------------------------------------------------------------- #
# Registry: observed-only refinement
# --------------------------------------------------------------------------- #
def test_repeat_observations_refine_and_shrink_sigma():
    reg = FixtureRegistry()
    first = reg.observe("knob", np.array([1.0, 0.0, 0.0]), sigma=0.2)
    assert first.outcome is Outcome.NEW
    second = reg.observe("knob", np.array([1.10, 0.0, 0.0]), sigma=0.2)
    assert second.outcome is Outcome.REFINED
    assert second.landmark is first.landmark
    assert second.landmark.n_obs == 2
    assert second.landmark.position[0] == pytest.approx(1.05)
    assert second.landmark.sigma == pytest.approx(0.2 / math.sqrt(2.0))
    assert second.landmark.sigma < 0.2
    assert len(reg.landmarks) == 1


def test_low_sigma_observation_dominates_the_weighted_update():
    reg = FixtureRegistry()
    reg.observe("knob", np.array([1.0, 0.0, 0.0]), sigma=0.5)
    assoc = reg.observe("knob", np.array([1.20, 0.0, 0.0]), sigma=0.05)
    # w = 1/sigma^2 -> 4 vs 400, so the sharp measurement carries 99% of the weight.
    w_old, w_new = 1.0 / 0.5 ** 2, 1.0 / 0.05 ** 2
    expected = (1.0 * w_old + 1.20 * w_new) / (w_old + w_new)
    assert assoc.landmark.position[0] == pytest.approx(expected)
    assert assoc.landmark.position[0] > 1.19
    assert assoc.landmark.sigma == pytest.approx(math.sqrt(1.0 / (w_old + w_new)))


def test_out_of_range_observation_is_rejected_before_association():
    reg = _registry_with_cad(max_range=6.0)
    assoc = reg.observe("valve_wheel", CAD_POS.copy(), sigma=0.2, range_m=9.0)
    assert assoc.outcome is Outcome.REJECTED
    assert assoc.landmark is None
    assert len(reg.landmarks) == 1
    assert reg.landmarks[0].n_obs == 0


def test_in_range_observation_is_not_rejected():
    reg = _registry_with_cad(max_range=6.0)
    assoc = reg.observe("valve_wheel", CAD_POS.copy(), sigma=0.2, range_m=5.9)
    assert assoc.outcome is Outcome.CONFIRMED


def test_patches_accumulate_on_the_landmark():
    reg = _registry_with_cad()
    reg.observe("valve_wheel", CAD_POS.copy(), sigma=0.2, patch="a.png")
    reg.observe("valve_wheel", CAD_POS.copy(), sigma=0.2, patch="b.png")
    assert reg.landmarks[0].patches == ["a.png", "b.png"]


# --------------------------------------------------------------------------- #
# Registry: correspondences and YAML
# --------------------------------------------------------------------------- #
def test_correspondences_excludes_observed_only_by_default():
    reg = _registry_with_cad()
    reg.observe("valve_wheel", CAD_POS.copy(), sigma=0.2)
    reg.observe("knob", np.array([-2.0, 0.0, 0.0]), sigma=0.2)
    ids = [lid for lid, _ in reg.correspondences()]
    assert ids == ["valve_wheel_cad_01"]
    ids_all = [lid for lid, _ in reg.correspondences(confirmed_only=False)]
    assert set(ids_all) == {"valve_wheel_cad_01", "knob_obs_001"}


def test_correspondences_skips_unobserved_and_flagged_cad_entries():
    reg = _registry_with_cad()
    assert reg.correspondences() == []          # never seen
    reg.observe("valve_wheel", CAD_POS + np.array([0.0, 0.35, 0.0]), sigma=0.2)
    assert reg.correspondences() == []          # seen, but flagged


def test_yaml_roundtrip_preserves_positions_origin_and_patches(tmp_path):
    reg = _registry_with_cad()
    reg.observe("valve_wheel", CAD_POS + np.array([0.02, 0.0, 0.0]), sigma=0.2, patch="p0.png")
    reg.observe("knob", np.array([-1.0, 0.25, 0.8]), sigma=0.3, patch="p1.png")

    out = tmp_path / "map_out.yaml"
    reg.save(out)

    reloaded = FixtureRegistry()
    reloaded.load_apriori(out)

    assert len(reloaded.landmarks) == len(reg.landmarks)
    for before, after in zip(reg.landmarks, reloaded.landmarks):
        assert after.id == before.id
        assert after.cls == before.cls
        assert after.origin is before.origin
        assert after.patches == before.patches
        assert after.n_obs == before.n_obs
        assert after.sigma == pytest.approx(before.sigma)
        np.testing.assert_allclose(after.position, before.position)

    knob = [lm for lm in reloaded.landmarks if lm.cls == "knob"][0]
    assert knob.origin is Origin.OBSERVED
    assert knob.patches == ["p1.png"]


def test_landmark_as_dict_is_plain_python():
    lm = Landmark(id="x", cls="knob", position=np.array([1.0, 2.0, 3.0]),
                  origin=Origin.APRIORI, sigma=0.01)
    d = lm.as_dict()
    assert d["position"] == [1.0, 2.0, 3.0]
    assert all(isinstance(v, float) for v in d["position"])
    assert d["origin"] == "apriori"
    np.testing.assert_allclose(Landmark.from_dict(d).position, lm.position)
