from glob import glob
from setuptools import setup

package_name = "rebotarm_bringup"
config_files = [
    "config/driver_params.yaml",
    "config/rebotarm_hardware.yaml",
]
urdf_files = [
    "description/urdf/reBot-DevArm_fixend.urdf",
    "description/urdf/reBot_B601_DM_with_gripper.urdf",
]

setup(
    name=package_name,
    version="0.3.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", config_files),
        (f"share/{package_name}/description/urdf", urdf_files),
        (f"share/{package_name}/description/meshes", glob("description/meshes/*")),
        (
            f"share/{package_name}/description/meshes_b601_gripper",
            glob("description/meshes_b601_gripper/*"),
        ),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="reBotArm Maintainers",
    maintainer_email="support@example.com",
    description="Launch, configuration, and description files for reBotArm.",
    license="Apache-2.0",
)
