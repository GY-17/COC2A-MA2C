#!/usr/bin/env bash
# ============================================================
# TABLE III (MACA-MA2C vs MIP) 复现脚本
# 用法: 在 Git Bash 里 cd 到项目根目录后执行:  bash run_table3.sh
# 说明: 每次运行前会自动把 config 的 num_customers 改成 n-1
#       (n=5->4, n=10->9, n=20->19), 因为数据集的
#       "Number of Nodes" 含 1 个 depot, 实际 customers = n-1。
# ============================================================
cd "E:/python_object/VRPD/2pre_reward_Entropy_annealing_maca2c_rolloutlen" || exit 1

CKPT="./VRPD/results/checkpoints/V1"   # 你的模型
SAVE="./benchmark_results/MACA2C3.5"

run_one () {
  local N=$1 NC=$2 SUF=$3 START=$4
  echo "=================================================="
  echo ">>> TABLE III : n=$N  (config num_customers=$NC, suffix=$SUF, start=$START)"
  echo "=================================================="
  # 自动修改 trucks_and_drones/config.py 的 num_customers
  python - "$NC" <<'PY'
import re, sys
p = "trucks_and_drones/config.py"
s = open(p, encoding='utf-8').read()
s = re.sub(r"('num_customers':\s*)\d+", r"\g<1>" + sys.argv[1], s, count=1)
open(p, 'w', encoding='utf-8').write(s)
print("config num_customers ->", sys.argv[1])
PY
  python batch_evaluate.py \
    --algorithm maca2c \
    --checkpoint_dir "$CKPT" \
    --episodes 3 \
    --instances 10 \
    --damages 0.1 0.3 0.5 \
    --prefix "singlecenter-" \
    --suffix "$SUF" \
    --save_dir "$SAVE" \
    --start_index "$START"
}

# n=5  : 1 depot + 4 customers, 数据集 singlecenter-1..10-n5.txt
run_one 5  4  "-n5.txt"  1
# n=10 : 1 depot + 9 customers, 数据集 singlecenter-51..60-n10.txt
run_one 10 9  "-n10.txt" 51
# n=20 : 1 depot + 19 customers, 数据集 singlecenter-61..70-n20.txt
run_one 20 19 "-n20.txt" 61

# ⚠️ 还原 config：TABLE III 临时把 num_customers 改成了 n-1 (4/9/19)，
#    跑完必须恢复成训练规模（V1 模型对应的 174 客户），否则后续动态评估
#    analyze_damage_dod 会误用 4 客户玩具实例，结果全错。
python - <<'PY'
import re
p = "trucks_and_drones/config.py"
s = open(p, encoding='utf-8').read()
s = re.sub(r"('num_customers':\s*)\d+", r"\g<1>174", s, count=1)
open(p, 'w', encoding='utf-8').write(s)
print("config num_customers -> 174 (restored)")
PY
