#!/usr/bin/env bash
#
# 批量运行一个 TSC 算法在多个场景下的测试, 结果保存到 results/。
#
# 每个 (junction, net, pattern) 组合都在独立的 python 进程中运行算法入口脚本
# (traditional -> run.py, rl -> eval.py), 输出写到
#   results/<algo>/<junction>_<env_name>/trip_info.xml (及 fcd_output.xml)。
#
# 为什么用独立进程: libsumo 是进程级单例, 在同一进程内连续构建多个不同路口的
# 环境会泄漏 lanearea 订阅, 触发 KeyError; 每个组合单独起进程可彻底避免。
# 也因此可以安全并行 (--jobs N), 各进程结果目录互不冲突。
#
# 本脚本只按退出码判定通过/失败; 指标汇总请用:
#   python scripts/analysis_tripinfo.py --csv results/summary.csv
#
# 用法:
#   conda activate tshub
#   # 一个算法测试完毕所有的场景
#   bash scripts/run_batch.sh --algo fixtime --jobs 10
#   # fixtime 跑全部 12 路口 x easy/normal x 5 pattern, 注入 event_1, 4 个并行
#   bash scripts/run_batch.sh --algo fixtime --event_name event_1 --jobs 4
#   # 只跑部分场景
#   bash scripts/run_batch.sh --algo maxpressure \
#       --junctions Beijing_Beihuan,Hongkong_YMT --nets normal --patterns low_density
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ALGO=""
JUNCTIONS=""
NETS="easy normal"
PATTERNS="low_density high_density fluctuating_commuter increasing_demand random_perturbation"
EVENT_NAME=""
JOBS=1

usage() {
  cat >&2 <<'EOF'
usage: run_batch.sh --algo <name> [options]
  --algo NAME         必填; fixtime|maxpressure|webster|sotl|presslight|attendlight|intellilight|unitsa
  --junctions LIST    逗号分隔的路口; 默认全部 12 个
  --nets LIST         逗号分隔的路网 (easy,normal); 默认两者
  --patterns LIST     逗号分隔的交通流模式; 默认全部 5 种
  --event_name NAME   注入的特殊事件集合 (如 event_1); 默认不注入
  --jobs N            并行进程数; 默认 1 (顺序执行)
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo)       ALGO="$2"; shift 2;;
    --junctions)  JUNCTIONS="${2//,/ }"; shift 2;;
    --nets)       NETS="${2//,/ }"; shift 2;;
    --patterns)   PATTERNS="${2//,/ }"; shift 2;;
    --event_name) EVENT_NAME="$2"; shift 2;;
    --jobs)       JOBS="$2"; shift 2;;
    -h|--help)    usage 0;;
    *) echo "unknown argument: $1" >&2; usage 2;;
  esac
done

[[ -z "$ALGO" ]] && { echo "error: --algo is required" >&2; usage 2; }
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "error: --jobs must be a positive integer" >&2; exit 2; }

# 算法 -> 入口脚本
case "$ALGO" in
  fixtime|maxpressure|webster|sotl)            ENTRY="tsc_algos/traditional/$ALGO/run.py";;
  presslight|attendlight|intellilight|unitsa)  ENTRY="tsc_algos/rl/$ALGO/eval.py";;
  *) echo "error: unknown algo '$ALGO'" >&2; exit 2;;
esac

# 默认跑全部 12 个路口
if [[ -z "$JUNCTIONS" ]]; then
  JUNCTIONS="$(python -c 'from junction_configs import AVAILABLE_JUNCTIONS; print(" ".join(AVAILABLE_JUNCTIONS))')"
fi

EVT=()
[[ -n "$EVENT_NAME" ]] && EVT=(--event_name "$EVENT_NAME")

RESULTFILE="$(mktemp)"
trap 'rm -f "$RESULTFILE"' EXIT

# 运行单个组合: 跑算法入口脚本, 按退出码记录结果 (一行写入 RESULTFILE, 原子追加)
run_one() {
  local j="$1" env_name="$2" err msg
  err="$(mktemp)"
  if python "$ROOT/$ENTRY" --junction "$j" --env_name "$env_name" "${EVT[@]}" \
        >/dev/null 2>"$err"; then
    printf '%s\t%s\t\n' "OK" "$j/$env_name" >>"$RESULTFILE"
    printf '[ OK   ] %s/%s\n' "$j" "$env_name"
  else
    msg="$(tail -n1 "$err" 2>/dev/null)"
    printf '%s\t%s\t%s\n' "FAIL" "$j/$env_name" "$msg" >>"$RESULTFILE"
    printf '[ FAIL ] %s/%s -> %s\n' "$j" "$env_name" "$msg"
  fi
  rm -f "$err"
}

n_total=0
echo "algo=$ALGO entry=$ENTRY event_name=${EVENT_NAME:-(none)} jobs=$JOBS"
echo

for j in $JUNCTIONS; do
  for net in $NETS; do
    for pat in $PATTERNS; do
      n_total=$((n_total + 1))
      if (( JOBS > 1 )); then
        run_one "$j" "${net}_${pat}" &
        while (( $(jobs -r | wc -l) >= JOBS )); do wait -n; done
      else
        run_one "$j" "${net}_${pat}"
      fi
    done
  done
done
wait

pass=$(grep -c $'^OK\t' "$RESULTFILE" || true)
fail=$(grep -c $'^FAIL\t' "$RESULTFILE" || true)

echo
echo "================ SUMMARY: OK=$pass FAIL=$fail / total=$n_total ================"
if (( fail > 0 )); then
  awk -F'\t' '$1=="FAIL"{printf "  %s: %s\n", $2, $3}' "$RESULTFILE"
fi
echo "metrics: python scripts/analysis_tripinfo.py --csv results/summary.csv"
exit $(( fail > 0 ? 1 : 0 ))
