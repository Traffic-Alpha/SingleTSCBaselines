#!/usr/bin/env bash
#
# 顺序训练一个 RL 算法在多个场景上的模型: 一个场景训练完成后再开始下一个。
# (训练资源占用高, 故只串行, 不并行。)
#
# 每个 (junction, net, pattern) 组合调用算法的 train.py:
#   python tsc_algos/rl/<algo>/train.py --junction J --env_name <net>_<pattern> [透传参数...]
# checkpoint / tensorboard / log 由各 train.py 自行保存 (位于算法目录下, 已 gitignore)。
#
# `--` 之后的所有参数都会原样透传给 train.py, 用于指定训练超参 (各算法可用参数不同),
# 例如 --total_timesteps / --num_envs / --vec_env / --seed 等。
#
# 用法:
#   conda activate tshub
#   # presslight 在全部 12 路口 x easy/normal x 5 pattern 上依次训练
#   bash scripts/train_batch.sh --algo presslight -- --total_timesteps 300000 --num_envs 20 --history_len 5 --vec_env subproc
#   # unitsa 在全部 12 路口 x easy/normal x 5 pattern 上依次训练
#   bash scripts/train_batch.sh --algo unitsa -- --total_timesteps 300000 --num_envs 20 --history_len 5 --vec_env subproc
#   # 只训练部分场景
#   bash scripts/train_batch.sh --algo unitsa --junctions Beijing_Beihuan --nets normal \
#       --patterns fluctuating_commuter -- --total_timesteps 500000
#   # 只打印将要执行的命令, 不实际训练
#   bash scripts/train_batch.sh --algo presslight --dry-run
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ALGO=""
JUNCTIONS=""
NETS="easy normal"
PATTERNS="low_density high_density fluctuating_commuter increasing_demand random_perturbation"
DRY_RUN=0
EXTRA=()  # 透传给 train.py 的参数 (位于 -- 之后)

usage() {
  cat >&2 <<'EOF'
usage: train_batch.sh --algo <name> [options] [-- <train.py 透传参数>]
  --algo NAME        必填; presslight|attendlight|intellilight|unitsa
  --junctions LIST   逗号分隔的路口; 默认全部 12 个
  --nets LIST        逗号分隔的路网 (easy,normal); 默认两者
  --patterns LIST    逗号分隔的交通流模式; 默认全部 5 种
  --dry-run          只打印将执行的命令, 不实际训练
  -- ...             其后参数原样透传给 train.py (训练超参)
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo)      ALGO="$2"; shift 2;;
    --junctions) JUNCTIONS="${2//,/ }"; shift 2;;
    --nets)      NETS="${2//,/ }"; shift 2;;
    --patterns)  PATTERNS="${2//,/ }"; shift 2;;
    --dry-run)   DRY_RUN=1; shift;;
    -h|--help)   usage 0;;
    --)          shift; EXTRA=("$@"); break;;
    *) echo "unknown argument: $1" >&2; usage 2;;
  esac
done

[[ -z "$ALGO" ]] && { echo "error: --algo is required" >&2; usage 2; }
case "$ALGO" in
  presslight|attendlight|intellilight|unitsa) ENTRY="tsc_algos/rl/$ALGO/train.py";;
  *) echo "error: --algo must be an RL algo (presslight|attendlight|intellilight|unitsa)" >&2; exit 2;;
esac

# 默认全部 12 个路口
if [[ -z "$JUNCTIONS" ]]; then
  JUNCTIONS="$(python -c 'from junction_configs import AVAILABLE_JUNCTIONS; print(" ".join(AVAILABLE_JUNCTIONS))')"
fi

# 组合列表
combos=()
for j in $JUNCTIONS; do
  for net in $NETS; do
    for pat in $PATTERNS; do
      combos+=("$j ${net}_${pat}")
    done
  done
done

echo "algo=$ALGO entry=$ENTRY scenarios=${#combos[@]} extra=[${EXTRA[*]:-}]"
echo

idx=0
pass=0; fail=0; failed=()
for combo in "${combos[@]}"; do
  read -r j env_name <<<"$combo"
  idx=$((idx + 1))
  echo "================ [$idx/${#combos[@]}] $ALGO :: $j/$env_name ================"
  cmd=(python "$ROOT/$ENTRY" --junction "$j" --env_name "$env_name" "${EXTRA[@]}")
  if (( DRY_RUN )); then
    printf 'DRY-RUN: %s\n' "${cmd[*]}"
    continue
  fi
  if "${cmd[@]}"; then
    echo "[ DONE ] $j/$env_name"
    pass=$((pass + 1))
  else
    echo "[ FAIL ] $j/$env_name (exit $?)"
    fail=$((fail + 1))
    failed+=("$j/$env_name")
  fi
  echo
done

(( DRY_RUN )) && exit 0

echo "================ SUMMARY: DONE=$pass FAIL=$fail / total=${#combos[@]} ================"
if (( fail > 0 )); then
  printf '  FAIL: %s\n' "${failed[@]}"
fi
exit $(( fail > 0 ? 1 : 0 ))
