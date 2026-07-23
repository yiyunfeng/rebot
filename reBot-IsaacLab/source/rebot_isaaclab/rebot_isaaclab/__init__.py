"""reBot Isaac Lab 顶层包。

任务注册位于 :mod:`rebot_isaaclab.tasks`，只能在 ``AppLauncher`` 启动后导入。
顶层包故意不自动导入任务，使 :mod:`rebot_isaaclab.metrics` 等纯 Python 工具
可以在没有 Kit、PhysX 和 GPU 的单元测试中独立使用。
"""
