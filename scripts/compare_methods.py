'''
@Author: WANG Maonan
@Description: 对比不同 TSC 方法在全部场景下的 TripInfo 指标。

结果目录结构 (由 output_utils 约定):
    results/<method>/<junction>_<net>_<pattern>/trip_info.xml

120 个场景 (12 路口 x 2 路网 x 5 模式) 不适合一个场景一张表, 本脚本按 **路口分组**:
每个路口一张表, 行 = 10 个 env (easy/normal x 5 pattern), 列 = 各方法, 每行最优值标 *。
这样 12 张表即可覆盖全部 120 个场景, 直观对比。

指标 (越小越优): travel_time(duration) / waiting_time(waitingTime) / time_loss(timeLoss)

用法:
    python scripts/compare_methods.py                       # 默认指标 travel_time
    python scripts/compare_methods.py --metric all          # 三个指标都打印
    python scripts/compare_methods.py --metric waiting_time
    python scripts/compare_methods.py --csv-dir results/    # 同时导出宽表 CSV
'''
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tshub.sumo_tools.analysis_output.tripinfo_analysis import TripInfoAnalysis

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# 指标: TripInfo 列名 -> 展示名 (均为越小越优)
METRICS = {
    "duration": "travel_time",
    "waitingTime": "waiting_time",
    "timeLoss": "time_loss",
}
NETS = ["easy", "normal"]
PATTERNS = ["low_density", "high_density", "fluctuating_commuter",
            "increasing_demand", "random_perturbation"]


def split_scenario(dirname: str):
    """'Beijing_Beihuan_normal_low_density' -> (junction, net, pattern)。
    路口名含下划线, 故按已知 net/pattern 后缀反向切分。"""
    for net in NETS:
        for pat in PATTERNS:
            suffix = f"{net}_{pat}"
            if dirname.endswith(suffix) and len(dirname) > len(suffix) + 1:
                return dirname[: -len(suffix) - 1], net, pat
    return dirname, "", ""  # 兜底


def collect(results_dir: Path) -> pd.DataFrame:
    """扫描所有 method/scenario, 返回 long 表。"""
    records = []
    for trip in sorted(results_dir.glob("*/*/trip_info.xml")):
        method = trip.parent.parent.name
        junction, net, pattern = split_scenario(trip.parent.name)
        parser = TripInfoAnalysis(str(trip))
        if parser.df is None or parser.df.empty:
            continue
        stats = parser.calculate_multiple_stats(metrics=list(METRICS.keys()))
        rec = {
            "method": method, "junction": junction, "net": net, "pattern": pattern,
            "env": f"{net}_{pattern}", "num_vehicles": len(parser.df),
        }
        for raw, nice in METRICS.items():
            rec[nice] = stats[raw]["mean"] if (stats and raw in stats) else float("nan")
        records.append(rec)
    return pd.DataFrame.from_records(records)


def fmt_table_with_best(pivot: pd.DataFrame) -> pd.DataFrame:
    """把数值格式化为字符串, 每行最优 (最小) 值加 * 标记。"""
    out = pivot.copy().astype(object)
    for idx, row in pivot.iterrows():
        if row.notna().any():
            best_col = row.idxmin()
        else:
            best_col = None
        for col in pivot.columns:
            v = row[col]
            if pd.isna(v):
                out.at[idx, col] = "-"
            else:
                out.at[idx, col] = f"{v:.2f}" + ("*" if col == best_col else " ")
    return out


def print_per_junction(df: pd.DataFrame, metric: str):
    print(f"\n{'#' * 80}\n# Metric: {metric}  (lower is better, * = best per scenario)\n{'#' * 80}")
    for junction in sorted(df["junction"].unique()):
        sub = df[df["junction"] == junction]
        pivot = sub.pivot_table(index="env", columns="method", values=metric)
        # env 行排序: easy_* 在前, normal_* 在后, 按 pattern 顺序
        order = [f"{n}_{p}" for n in NETS for p in PATTERNS if f"{n}_{p}" in pivot.index]
        pivot = pivot.reindex(order)
        print(f"\n=== {junction} ===")
        print(fmt_table_with_best(pivot).to_string())


def print_ranking(df: pd.DataFrame):
    """总体排名: 每个方法在全部场景上的平均排名(1=最好) 与 夺冠次数。"""
    print(f"\n{'#' * 80}\n# Overall ranking across all scenarios (mean rank, lower=better; wins=#best)\n{'#' * 80}")
    rows = {}
    methods = sorted(df["method"].unique())
    for nice in METRICS.values():
        wide = df.pivot_table(index=["junction", "env"], columns="method", values=nice)
        ranks = wide.rank(axis=1, method="min")          # 每个场景内排名
        mean_rank = ranks.mean()
        wins = (wide.eq(wide.min(axis=1), axis=0)).sum()  # 取得最小值的次数
        for m in methods:
            rows.setdefault(m, {})[f"{nice}_rank"] = mean_rank.get(m, float("nan"))
            rows.setdefault(m, {})[f"{nice}_wins"] = int(wins.get(m, 0))
    table = pd.DataFrame.from_dict(rows, orient="index")
    # 按 travel_time 平均排名排序
    table = table.sort_values("travel_time_rank")
    show = table.copy()
    for c in show.columns:
        if c.endswith("_rank"):
            show[c] = show[c].map(lambda v: f"{v:.2f}")
    print("\n" + show.to_string())
    n_scen = df.groupby("method").size().min()
    print(f"\n(每个方法覆盖 {n_scen} 个场景; 共 {df['junction'].nunique()} 路口 x {len(NETS)} 路网 x {len(PATTERNS)} 模式)")


def _md_table(pivot: pd.DataFrame, methods: list) -> str:
    """把 env x method 的 pivot 渲染成 markdown 表, 每行最优 (最小) 值加粗。"""
    lines = ["| scenario | " + " | ".join(methods) + " |",
             "|" + "---|" * (len(methods) + 1)]
    for env in pivot.index:
        row = pivot.loc[env]
        best = row.idxmin() if row.notna().any() else None
        cells = []
        for m in methods:
            v = row.get(m, float("nan"))
            if pd.isna(v):
                cells.append("-")
            else:
                cells.append(f"**{v:.2f}**" if m == best else f"{v:.2f}")
        lines.append(f"| {env} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_markdown(df: pd.DataFrame, metrics: list) -> str:
    """生成 README 用的 markdown 报告: 总体排名 + 每个路口的详细对比表。"""
    methods = sorted(df["method"].unique())
    n_junc = df["junction"].nunique()
    out = []
    out.append("## Benchmark Results\n")
    out.append(
        f"Evaluated across **{len(df) // len(methods)} scenarios** "
        f"({n_junc} intersections × {len(NETS)} networks × {len(PATTERNS)} traffic patterns). "
        "Metrics are SUMO `tripinfo` means in seconds (**lower is better**); "
        "in each per-intersection table the **best** method per scenario is in bold.\n"
    )

    # ---- 总体排名 ----
    out.append("### Overall Ranking\n")
    out.append("Mean rank (1 = best) and number of scenarios won, across all scenarios.\n")
    header = ["Method"]
    for nice in metrics:
        header += [f"{nice} rank", f"{nice} wins"]
    rows = {}
    for nice in metrics:
        wide = df.pivot_table(index=["junction", "env"], columns="method", values=nice)
        ranks = wide.rank(axis=1, method="min")
        wins = wide.eq(wide.min(axis=1), axis=0).sum()
        for m in methods:
            rows.setdefault(m, {})[f"{nice} rank"] = ranks.mean().get(m, float("nan"))
            rows.setdefault(m, {})[f"{nice} wins"] = int(wins.get(m, 0))
    rank_df = pd.DataFrame.from_dict(rows, orient="index").sort_values(f"{metrics[0]} rank")
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for m, r in rank_df.iterrows():
        cells = [m]
        for nice in metrics:
            cells.append(f"{r[f'{nice} rank']:.2f}")
            cells.append(str(int(r[f'{nice} wins'])))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    # ---- 每个路口的详细结果 ----
    out.append("### Per-Intersection Detail\n")
    out.append("<details>\n<summary>展开查看每个路口的详细对比 (click to expand)</summary>\n")
    for junction in sorted(df["junction"].unique()):
        sub = df[df["junction"] == junction]
        out.append(f"\n#### {junction}\n")
        for nice in metrics:
            pivot = sub.pivot_table(index="env", columns="method", values=nice)
            order = [f"{n}_{p}" for n in NETS for p in PATTERNS if f"{n}_{p}" in pivot.index]
            pivot = pivot.reindex(order)
            out.append(f"*{nice} (s)*\n")
            out.append(_md_table(pivot, methods))
            out.append("")
    out.append("</details>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="对比不同 TSC 方法在全部场景下的 TripInfo 指标")
    ap.add_argument("--metric", default="travel_time",
                    choices=list(METRICS.values()) + ["all"],
                    help="按路口分组打印的指标; all 表示三个都打印")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR), help="results 目录")
    ap.add_argument("--csv-dir", default=None,
                    help="若指定, 导出每个指标的宽表 CSV (scenario x method) 到该目录")
    ap.add_argument("--markdown", default=None,
                    help="若指定, 生成 README 用的 markdown 报告到该文件")
    ap.add_argument("--md-metrics", default="travel_time,waiting_time",
                    help="markdown 报告使用的指标 (逗号分隔)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    df = collect(results_dir)
    if df.empty:
        print(f"未在 {results_dir} 找到任何 trip_info.xml 数据。")
        return

    print(f"方法: {sorted(df['method'].unique())}")
    print(f"场景数 (各方法): {dict(df.groupby('method').size())}")

    if args.markdown:
        md_metrics = [m.strip() for m in args.md_metrics.split(",") if m.strip()]
        Path(args.markdown).write_text(build_markdown(df, md_metrics) + "\n", encoding="utf-8")
        print(f"\nMarkdown 报告已写入: {args.markdown} (指标: {md_metrics})")
        return

    metrics = list(METRICS.values()) if args.metric == "all" else [args.metric]
    for metric in metrics:
        print_per_junction(df, metric)

    print_ranking(df)

    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        # long 表
        df.sort_values(["junction", "env", "method"]).to_csv(csv_dir / "comparison_long.csv", index=False, float_format="%.2f")
        # 每个指标一个宽表 (行=scenario, 列=method)
        for nice in METRICS.values():
            wide = df.pivot_table(index=["junction", "env"], columns="method", values=nice)
            wide.to_csv(csv_dir / f"comparison_{nice}.csv", float_format="%.2f")
        print(f"\nCSV 已导出至: {csv_dir} (comparison_long.csv + comparison_<metric>.csv)")


if __name__ == "__main__":
    main()
