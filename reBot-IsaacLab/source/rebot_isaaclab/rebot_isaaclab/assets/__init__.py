"""对外导出任务需要的机器人配置和香蕉 USD 路径。"""

# 调用方只从 assets 包导入公共对象，不依赖 rebot_dm.py 的内部路径变量。
from .rebot_dm import BANANA_USD_PATH, REBOT_DM_CFG

# 明确公共 API，避免 wildcard import 暴露 sim_utils 等实现细节。
__all__ = ["BANANA_USD_PATH", "REBOT_DM_CFG"]
