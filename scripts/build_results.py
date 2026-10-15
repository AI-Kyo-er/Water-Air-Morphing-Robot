"""从实际基准 CSV 自动生成结果表和科研图，不手写性能数字。"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "experiments/control/allocation_benchmark"
FUSION = ROOT / "experiments/perception/reliability_benchmark"
LABELS = {"pinv_clip": "伪逆＋限幅", "common_tilt": "共倾转约束分配", "proposed": "本文方法",
          "no_rate": "去除速率约束", "no_health": "去除效能信息",
          "single_acoustic": "单声学", "fixed_fusion": "固定融合",
          "no_quality": "去除质量权重", "no_temporal": "去除时间迟滞"}
CASES = {"nominal": "常规机动", "yaw_maneuver": "偏航机动", "thrust_loss": "推力效能下降",
         "fast_command": "快速指令变化", "medium_transition": "介质模式切换"}
COLORS = ["#738795", "#dfaa59", "#179b8a", "#9474b3", "#dc7b72"]


def read_rows(path: Path) -> list[dict]:
    """读取带列名的结果 CSV。"""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_plotting() -> None:
    """选择可用中文字体和统一论文图样式。"""
    windows_font = Path("C:/Windows/Fonts/msyh.ttc")
    if windows_font.exists():
        font_manager.fontManager.addfont(str(windows_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(windows_font)).get_name()
    else:
        plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "font.size": 11,
                         "axes.labelcolor": "#314c5b", "text.color": "#173342",
                         "svg.fonttype": "none", "figure.facecolor": "white"})


def save_plot(fig, destination: Path) -> None:
    """输出可编辑 SVG 与预览 PNG，图注保留结果条件。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.02, 0.015, "潜空智航 · 控制分配与可信感知结果", fontsize=8,
             color="#6c7e87")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    svg_path = destination.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in
                        svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8")
    fig.savefig(destination.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def control_results() -> None:
    """聚合全部种子，生成对比和执行量响应。"""
    rows = read_rows(CONTROL / "data/metrics.csv")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["method"])].append(row)
    cases = list(dict.fromkeys(row["case"] for row in rows))
    methods = ["pinv_clip", "common_tilt", "proposed"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(cases))
    for i, method in enumerate(methods):
        samples = [[float(r["normalized_rmse"]) for r in groups[(case, method)]] for case in cases]
        axes[0].bar(x + (i-1)*0.24, [np.mean(v) for v in samples], 0.24,
                    yerr=[np.std(v) for v in samples], color=COLORS[i], label=LABELS[method], capsize=3)
    axes[0].set_xticks(x, [CASES[case] for case in cases], rotation=15)
    axes[0].set_ylabel("加权归一化分配误差 RMSE")
    axes[0].set_title("同输入与同执行器约束下的分配对比")
    axes[0].legend(fontsize=9)
    traces = read_rows(CONTROL / "data/traces.csv")
    for i, method in enumerate(methods):
        selected = [r for r in traces if r["case"] == "yaw_maneuver" and r["method"] == method]
        axes[1].plot([float(r["time_s"]) for r in selected],
                     [float(r["achieved_tz_nm"]) for r in selected], color=COLORS[i],
                     label=LABELS[method], linewidth=1.7)
    selected = [r for r in traces if r["case"] == "yaw_maneuver" and r["method"] == "proposed"]
    axes[1].plot([float(r["time_s"]) for r in selected], [float(r["target_tz_nm"]) for r in selected],
                 "--", color="#173342", label="期望偏航力矩", linewidth=1)
    axes[1].set(xlabel="时间 / s", ylabel="偏航力矩 / N·m", title="独立倾转改善偏航力矩响应")
    axes[1].legend(fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_plot(fig, CONTROL / "figures/allocation_comparison")
    table = ["# 约束控制分配结果", "", "结果由 `scripts/build_results.py` 从全部种子记录计算。",
             "数值为均值 ± 标准差；归一化误差使用集中配置中的力/力矩尺度与轴优先级。", "",
             "| 场景 | 方法 | 归一化 RMSE | 速率越界比例 | 求解耗时 P95 / ms |", "|---|---|---:|---:|---:|"]
    for case in cases:
        for method in list(dict.fromkeys(r["method"] for r in rows)):
            records = groups[(case, method)]
            values = [float(r["normalized_rmse"]) for r in records]
            violations = np.mean([float(r["rate_violation_fraction"]) for r in records])
            runtime = np.mean([float(r["solve_p95_ms"]) for r in records])
            table.append(f"| {CASES[case]} | {LABELS[method]} | {np.mean(values):.4f} ± {np.std(values):.4f} | {violations:.2%} | {runtime:.2f} |")
    table += ["", "去除速率约束的消融允许不可实现的单步跳变，只用于说明该约束的作用。",
              "效能下降场景给各完整方法相同的已知效能输入；去除效能信息用于单独检验该信息的价值。",
              "求解耗时在开发主机上测得，机载实时性需在目标平台复核。",
              "本结果评估分配层；整机闭环稳定性与入出水流体过程需在台架与水池中验证。"]
    (CONTROL / "results.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def fusion_results() -> None:
    """展示融合的准确率、概率质量和时间迟滞代价。"""
    rows = read_rows(FUSION / "data/metrics.csv")
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    methods = list(groups)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(methods))
    axes[0].bar(x - 0.18, [np.mean([float(r["f1"]) for r in groups[m]]) for m in methods],
                0.36, color="#179b8a", label="F1")
    axes[0].bar(x + 0.18, [np.mean([float(r["false_positive_rate"]) for r in groups[m]]) for m in methods],
                0.36, color="#dfaa59", label="虚警率")
    axes[0].set_xticks(x, [LABELS[m] for m in methods], rotation=18)
    axes[0].set(ylabel="比例", title="质量、新鲜度与时间判定的消融对比", ylim=(0, 1.08))
    axes[0].legend()
    trace = read_rows(FUSION / "data/traces.csv")
    for method, color in (("fixed_fusion", "#dfaa59"), ("proposed", "#179b8a")):
        selected = [r for r in trace if r["method"] == method]
        axes[1].plot([float(r["time_s"]) for r in selected], [float(r["posterior"]) for r in selected],
                     color=color, label=LABELS[method], linewidth=1)
    selected = [r for r in trace if r["method"] == "proposed"]
    tt = np.array([float(r["time_s"]) for r in selected])
    truth = np.array([float(r["target_present"]) for r in selected])
    axes[1].fill_between(tt, 0, 1, where=truth > 0, color="#d2e7ef", alpha=0.35, label="目标存在区间")
    axes[1].axvspan(25, 35, color="#f5e1ca", alpha=0.45, label="声学干扰区间")
    axes[1].set(xlabel="时间 / s", ylabel="目标存在概率", title="异常通道质量下降时抑制证据污染", ylim=(0, 1.05))
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_plot(fig, FUSION / "figures/reliability_comparison")
    table = ["# 可信感知融合结果", "", "独立校准种子与评估种子；包含质量下降、视觉缺失与声呐过期的时序场景。",
             "各模态质量由外部给定，可由声学谱平坦度、信噪比、磁噪声尺度等实测指标标定。", "",
             "| 方法 | 精确率 | 召回率 | F1 | 虚警率 | Brier 分数 | 判定切换次数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        avg = {key: np.mean([float(r[key]) for r in groups[method]]) for key in (
            "precision", "recall", "f1", "false_positive_rate", "brier_score", "decision_switches")}
        table.append(f"| {LABELS[method]} | {avg['precision']:.4f} | {avg['recall']:.4f} | {avg['f1']:.4f} | {avg['false_positive_rate']:.4f} | {avg['brier_score']:.4f} | {avg['decision_switches']:.1f} |")
    table += ["", "时间迟滞减少任务判定切换，但增加起止边界延迟；去除迟滞可能取得更高帧级 F1。",
              "质量权重的价值同时用 Brier 分数观察，不能只看二值判定结果。",
              "高斯似然与质量折扣的参数来自开发标定，实测概率校准随采集数据补充。"]
    (FUSION / "results.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def main() -> None:
    """生成完整结果包。"""
    configure_plotting()
    control_results()
    fusion_results()
    print("结果表和 SVG / PNG 图已生成。")


if __name__ == "__main__":
    main()
