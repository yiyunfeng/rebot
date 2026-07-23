"""rebot_isaaclab Python 包的 setuptools 安装入口。"""

from pathlib import Path

from setuptools import find_packages, setup


# 保留包根路径供后续读取 README/资源元数据；当前安装不依赖启动目录。
ROOT = Path(__file__).resolve().parent

# Isaac Lab 通过 editable install 加载该包；这里仅声明 Python 包元数据，
# Isaac Sim 扩展信息则由同目录 config/extension.toml 提供。
setup(
    name="rebot-isaaclab",
    version="0.1.0",
    description="Isaac Lab banana grasping tasks for the reBot Arm B601",
    # 自动包含 rebot_isaaclab 及其 tasks/assets/mdp 子包。
    packages=find_packages(),
    # 项目和 Isaac Sim 4.5 均使用 Python 3.10，不支持更早解释器。
    python_requires=">=3.10",
    # 禁止压缩为 egg zip，确保 USD/配置相关路径可按普通文件系统访问。
    zip_safe=False,
)
