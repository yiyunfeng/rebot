#!/usr/bin/env python3
"""检查项目内香蕉物理 USD 及其外观引用是否完整。

脚本名称保留为 ``build_banana_asset.py``，但当前物理 wrapper 已提交到仓库，
这里不会重新生成或覆盖 USD，只做快速静态检查。这样既避免误改资产，也能在
启动 Isaac Sim 前发现路径失效、defaultPrim 丢失或碰撞体缺失。
"""

from pathlib import Path

# 使用脚本自身位置确定路径，确保从任意工作目录调用都得到相同结果。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 外观模型继续复用 reBot-Isaacsim 中已有的香蕉资产，避免复制大体积二进制 USD。
SOURCE_USD = PROJECT_ROOT.parent / "reBot-Isaacsim" / "assets" / "banana" / "bananas_1k.usdc"
# wrapper 负责给原香蕉 mesh 补充刚体、质量和 convexDecomposition 碰撞。
OUTPUT_USD = PROJECT_ROOT / "assets" / "banana_physics.usda"


def main() -> None:
    """验证两个 USD 文件存在，并检查 wrapper 中的关键声明。"""

    # 分别检查源文件和 wrapper，报错时能明确知道是跨项目引用丢失还是本项目资产丢失。
    if not SOURCE_USD.is_file():
        raise FileNotFoundError(f"香蕉源模型不存在: {SOURCE_USD}")
    if not OUTPUT_USD.is_file():
        raise FileNotFoundError(f"香蕉物理资产不存在: {OUTPUT_USD}")

    # banana_physics.usda 是可读文本格式，因此无需启动 Isaac Sim/pxr 就能检查。
    text = OUTPUT_USD.read_text(encoding="utf-8")
    # 这些 token 对应资产可加载所需的最小结构；它不是完整 USD 语法验证器，
    # 但能快速捕获曾经出现过的 defaultPrim 和碰撞体缺失问题。
    required_tokens = (
        'defaultPrim = "Banana"',
        'prepend references = @../../reBot-Isaacsim/assets/banana/bananas_1k.usdc@</bananas/bananas_a>',
        '"PhysicsRigidBodyAPI"',
        '"PhysicsMassAPI"',
        '"PhysicsMeshCollisionAPI"',
        'uniform token physics:approximation = "convexDecomposition"',
        'float physxCollision:contactOffset = 0.002',
    )
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise RuntimeError(f"香蕉物理资产内容不完整，缺少: {missing}")

    print(f"[BananaAsset] 外观模型: {SOURCE_USD}")
    print(f"[BananaAsset] 物理资产: {OUTPUT_USD}")
    print("[BananaAsset] 碰撞体: original mesh + convexDecomposition")


if __name__ == "__main__":
    # 保留普通 Python 入口，便于 shell 脚本和 IDE 直接运行同一检查。
    main()
