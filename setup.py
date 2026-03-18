from setuptools import setup

setup(
    name="g1_nav",
    version="0.1.0",
    packages=["g1_nav"],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "waypoint_manager = g1_nav.waypoint_manager:main",
            "planner          = g1_nav.planner:main",
            "controller       = g1_nav.controller:main",
        ],
    },
)