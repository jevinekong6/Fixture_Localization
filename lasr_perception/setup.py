from glob import glob
import os

from setuptools import find_packages, setup

package_name = "lasr_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jevinekong",
    maintainer_email="jevinekong@gmail.com",
    description="YOLO fixture detections onto a vision_msgs/Detection2DArray topic.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_node = lasr_perception.yolo_node:main",
        ],
    },
)
