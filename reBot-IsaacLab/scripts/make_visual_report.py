#!/usr/bin/env python3
"""生成一页本地 HTML 报告，直观看懂 reBot IsaacLab 当前训练链路。

这个脚本不启动 Isaac Sim，也不依赖前端框架。它只读取当前项目里的
checkpoint、评估 JSON 和导出 metadata，把“训练了什么、结果在哪里、下一步
怎么跑”整理成一个浏览器能打开的静态页面。
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "banana_grasp_report.html"


def latest_file(pattern: str) -> Path | None:
    """按修改时间选择最新文件；不存在时返回 None，报告中显示“暂无”。"""

    files = sorted(PROJECT_ROOT.glob(pattern), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def read_json(path: Path | None) -> dict:
    """读取 JSON 文件；没有文件时返回空 dict，避免报告生成失败。"""

    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def short_path(path: Path | str | None) -> str:
    """把绝对路径压缩成相对项目路径，让页面更容易扫读。"""

    if path is None:
        return "暂无"
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fmt_rate(value: float | None) -> str:
    """成功率格式化为百分比。"""

    if value is None:
        return "暂无"
    return f"{value * 100:.1f}%"


def render() -> str:
    """拼出完整 HTML；样式内联，方便直接拷贝/打开。"""

    rgbd_eval_path = latest_file("results/eval_rgbd_*.json")
    rgbd_ckpt = latest_file("logs/rsl_rl/rebot_banana_grasp_return_rgbd/*/model_*.pt")
    isaacsim_teacher_data = PROJECT_ROOT / "data" / "rgbd_isaacsim_teacher_latest.pt"
    bc_policy = PROJECT_ROOT / "exported" / "rgbd_bc_policy.pt"
    export_meta_path = PROJECT_ROOT / "exported" / "rgbd_policy_latest.json"

    rgbd_eval = read_json(rgbd_eval_path)
    export_meta = read_json(export_meta_path if export_meta_path.exists() else None)

    rgbd_rate = rgbd_eval.get("success_rate")
    obs_dim = export_meta.get("observation_dim", 16405)
    proprio_size = export_meta.get("proprio_size", 21)
    image_size = obs_dim - proprio_size

    pipeline = [
        ("1", "香蕉资产", "原香蕉 USD + convexDecomposition 碰撞"),
        ("2", "并行环境", "机械臂、桌面、随机香蕉和摩擦"),
        ("3", "腕部 RGB-D", "64x64 RGB-D + 相机/光照误差"),
        ("4", "IsaacSim Teacher", "复用已验证传统抓取器生成示教动作"),
        ("5", "BC 预训练", "先学会接近、闭爪、抬升并返回 ready"),
        ("6", "PPO 微调", "只优化抓取、返回和稳定保持"),
        ("7", "成功率评估", "统计抓取返回成功率和置信区间"),
        ("8", "TorchScript", "导出给后续真机部署加载"),
    ]

    pipeline_html = "\n".join(
        f"""
        <div class="step">
          <div class="badge">{num}</div>
          <h3>{title}</h3>
          <p>{desc}</p>
        </div>
        """
        for num, title, desc in pipeline
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>reBot Banana Grasp Report</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #20242a;
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 32px 28px 48px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 30px; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; }}
    p {{ line-height: 1.55; }}
    code {{
      background: #eef1f5;
      padding: 2px 5px;
      border-radius: 4px;
    }}
    .muted {{ color: #667085; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-top: 18px;
    }}
    .card {{
      background: white;
      border: 1px solid #e2e6ea;
      border-radius: 8px;
      padding: 16px;
    }}
    .metric {{
      font-size: 34px;
      font-weight: 700;
      margin: 6px 0;
    }}
    .pipeline {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }}
    .step {{
      background: white;
      border: 1px solid #e2e6ea;
      border-radius: 8px;
      padding: 12px;
      min-height: 116px;
    }}
    .badge {{
      width: 26px;
      height: 26px;
      border-radius: 50%;
      background: #1f6feb;
      color: white;
      display: grid;
      place-items: center;
      font-weight: 700;
    }}
    .step h3 {{ margin: 10px 0 4px; font-size: 15px; }}
    .step p {{ margin: 0; color: #667085; font-size: 13px; }}
    .bar {{
      display: flex;
      height: 38px;
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid #d0d7de;
      background: white;
    }}
    .bar .proprio {{
      width: {proprio_size / obs_dim * 100:.2f}%;
      min-width: 58px;
      background: #54aeff;
    }}
    .bar .rgbd {{
      width: {image_size / obs_dim * 100:.2f}%;
      background: #ffd33d;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      margin-top: 8px;
      color: #667085;
      font-size: 14px;
    }}
    .dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 2px;
      margin-right: 6px;
      vertical-align: middle;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid #e2e6ea;
      border-radius: 8px;
      overflow: hidden;
    }}
    .table th, .table td {{
      padding: 10px 12px;
      border-bottom: 1px solid #eaeef2;
      text-align: left;
    }}
    .table tr:last-child td {{ border-bottom: 0; }}
    .cmd {{
      background: #111827;
      color: #f9fafb;
      padding: 12px;
      border-radius: 8px;
      overflow-x: auto;
      white-space: pre-wrap;
    }}
    @media (max-width: 900px) {{
      .grid, .pipeline {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>reBot IsaacLab 香蕉抓取可视化报告</h1>
  <p class="muted">生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}。这页只展示当前文件状态，不会启动仿真或训练。</p>

  <section class="grid">
    <div class="card">
      <div class="muted">RGB-D 策略成功率</div>
      <div class="metric">{fmt_rate(rgbd_rate)}</div>
      <div class="muted">评估文件：{html.escape(short_path(rgbd_eval_path))}</div>
    </div>
    <div class="card">
      <div class="muted">最新 checkpoint</div>
      <div class="metric">{'已生成' if rgbd_ckpt else '暂无'}</div>
      <div class="muted">{html.escape(short_path(rgbd_ckpt))}</div>
    </div>
    <div class="card">
      <div class="muted">部署模型</div>
      <div class="metric">{'已导出' if export_meta else '暂无'}</div>
      <div class="muted">{html.escape(short_path(export_meta.get("torchscript")))}</div>
    </div>
  </section>

  <h2>整体流程</h2>
  <section class="pipeline">{pipeline_html}</section>

  <h2>模仿学习链路状态</h2>
  <table class="table">
    <tr><th>阶段</th><th>状态</th><th>文件</th></tr>
    <tr><td>IsaacSim Teacher 数据</td><td>{'已生成' if isaacsim_teacher_data.exists() else '暂无'}</td><td><code>{html.escape(short_path(isaacsim_teacher_data if isaacsim_teacher_data.exists() else None))}</code></td></tr>
    <tr><td>BC 初始化权重</td><td>{'已生成' if bc_policy.exists() else '暂无'}</td><td><code>{html.escape(short_path(bc_policy if bc_policy.exists() else None))}</code></td></tr>
    <tr><td>PPO checkpoint</td><td>{'已生成' if rgbd_ckpt else '暂无'}</td><td><code>{html.escape(short_path(rgbd_ckpt))}</code></td></tr>
  </table>

  <h2>RGB-D 策略到底看什么</h2>
  <p>一次策略输入是 <code>{obs_dim}</code> 维：前 <code>{proprio_size}</code> 维是机器人本体状态，后 <code>{image_size}</code> 维是展平后的 <code>64x64x4</code> RGB-D 图像。</p>
  <div class="bar"><div class="proprio"></div><div class="rgbd"></div></div>
  <div class="legend">
    <span><span class="dot" style="background:#54aeff"></span>本体状态：joint_pos7 + joint_vel7 + last_action7</span>
    <span><span class="dot" style="background:#ffd33d"></span>RGB-D：RGB 三通道 + depth 一通道</span>
  </div>

  <h2>动作含义</h2>
  <table class="table">
    <tr><th>动作维度</th><th>含义</th><th>进入 Isaac Lab 后做什么</th></tr>
    <tr><td>0-2</td><td>末端 xyz 相对位移</td><td>Differential IK 转成 joint1-6 目标</td></tr>
    <tr><td>3-5</td><td>末端 roll/pitch/yaw 相对旋转</td><td>Differential IK 控制夹爪姿态</td></tr>
    <tr><td>6</td><td>夹爪开/关</td><td>控制 left_finger，右指由 mimic 跟随</td></tr>
  </table>

  <h2>当前文件</h2>
  <table class="table">
    <tr><th>类型</th><th>路径</th></tr>
    <tr><td>最新 RGB-D checkpoint</td><td><code>{html.escape(short_path(rgbd_ckpt))}</code></td></tr>
    <tr><td>RGB-D 导出 metadata</td><td><code>{html.escape(short_path(export_meta_path if export_meta else None))}</code></td></tr>
    <tr><td>RGB-D TorchScript</td><td><code>{html.escape(short_path(export_meta.get("torchscript")))}</code></td></tr>
  </table>

  <h2>最常用命令</h2>
<div class="cmd">cd {PROJECT_ROOT}
./run.sh build-asset
./run.sh collect-teacher
./run.sh train-bc
./run.sh train
REBOT_RGBD_RESUME=1 REBOT_RGBD_NUM_ENVS=16 REBOT_RGBD_ITERATIONS=500 ./run.sh train
REBOT_RGBD_EVAL_NUM_ENVS=1 REBOT_RGBD_EVAL_EPISODES=1 ./run.sh evaluate
./run.sh export
./run.sh report
./run.sh real-dry-run
./run.sh test
REBOT_REAL_POLICY_ENABLE=1 ./run.sh real-execute
./run.sh watch</div>
</main>
</body>
</html>
"""


def main() -> None:
    """写出报告并打印路径。"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render(), encoding="utf-8")
    print(f"[VisualReport] {REPORT_PATH}")


if __name__ == "__main__":
    main()
