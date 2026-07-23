"""导入项目任务模块，触发对应的 Gymnasium 注册。"""

# banana_lift 的导入副作用是执行 gym.register；任务脚本启动 Kit 后显式导入本包。
from . import banana_lift  # noqa: F401
