'''
@Author: WANG Maonan
@Description: 清理 RL eval 的 SUMO 输出文件。

RL 评估使用 SB3 的 DummyVecEnv, 会触发两次 env.reset(): 第一次启动 SUMO 后未步进
就被关闭, 写出一个空的 <name>.xml; 真正的评估是第二次, 由于 <name>.xml 已存在,
SUMO 自动改名写到 <name>_1.xml。结果每个场景目录会出现 4 个文件:
    trip_info.xml   (空, 0 车辆, 无用)      trip_info_1.xml  (真实结果)
    fcd_output.xml  (空, 无用)              fcd_output_1.xml (真实结果)

本脚本将每个 <stem>_1.xml 重命名为 <stem>.xml (覆盖掉空的无后缀文件), 这样:
  - 真实结果落到 trip_info.xml / fcd_output.xml, 与 analysis_tripinfo.py 约定一致;
  - 传统算法目录只有 trip_info.xml (没有 _1), 不会被改动, 安全。

用法:
    python scripts/clean_rl_output.py                       # 清理 results/ 下所有
    python scripts/clean_rl_output.py results/unitsa        # 只清理某算法
    python scripts/clean_rl_output.py --dry-run             # 只打印将执行的操作
'''
import argparse
from pathlib import Path

SUFFIX = "_1.xml"


def clean_dir(root: Path, dry_run: bool = False) -> int:
    """将 root 下所有 *_1.xml 重命名为去掉 _1 的同名文件, 返回处理数量。"""
    count = 0
    for src in sorted(root.rglob(f"*{SUFFIX}")):
        dst = src.with_name(src.name[: -len(SUFFIX)] + ".xml")
        action = "DRY-RUN" if dry_run else "rename "
        print(f"[{action}] {src.relative_to(root)} -> {dst.name}")
        if not dry_run:
            src.replace(dst)  # 原子替换, 覆盖掉空的无后缀文件
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description="清理 RL eval 的 SUMO 输出 (去掉 _1 后缀, 删除空文件)")
    default_root = Path(__file__).resolve().parents[1] / "results"
    ap.add_argument("root", nargs="?", default=str(default_root),
                    help="要清理的目录 (默认: 仓库根的 results/)")
    ap.add_argument("--dry-run", action="store_true", help="只打印, 不实际修改")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        ap.error(f"目录不存在: {root}")

    n = clean_dir(root, dry_run=args.dry_run)
    verb = "将处理" if args.dry_run else "已处理"
    print(f"\n{verb} {n} 个 *{SUFFIX} 文件 (位于 {root}).")


if __name__ == "__main__":
    main()
