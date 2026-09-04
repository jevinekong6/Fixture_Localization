from glob import glob
import os

from setuptools import find_packages, setup

package_name = "fixture_map_test"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "scripts"), glob("scripts/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jevinekong",
    maintainer_email="jevinekong@gmail.com",
    description=("YOLO fixture detections to map-frame landmarks with uncertainty, "
                 "image patches and RViz markers. Tripod-stage testing."),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fixture_map_node = fixture_map_test.fixture_map_node:main",
        ],
    },
)
