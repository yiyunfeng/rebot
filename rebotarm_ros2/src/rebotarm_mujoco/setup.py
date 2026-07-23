from glob import glob
from setuptools import setup

package_name = "rebotarm_mujoco"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    package_dir={package_name: "src"},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*")),
        (f"share/{package_name}/models", glob("models/*.xml")),
        (f"share/{package_name}/models/rebot_gripper_assets", glob("models/rebot_gripper_assets/*")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yyf",
    maintainer_email="yyf@example.com",
    description="MuJoCo simulation package for reBot Arm B601 DM.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mujoco_sim_node = rebotarm_mujoco.mujoco_sim_node:main",
        ],
    },
)
