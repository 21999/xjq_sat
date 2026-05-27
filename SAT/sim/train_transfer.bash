#!/usr/bin/env bash
set -euo pipefail

# task list
tasks=(
  dexart_bucket
  dexart_faucet
  dexart_laptop
  dexart_toilet
)

# additional info
info="v2.2.8-sat-transfer"
# gpu ids
gpu_device_ids=(0 1 2 3)
alg_name="sat"

num_tasks=${#tasks[@]}
num_gpus=${#gpu_device_ids[@]}


# copy template to avoid issues with modifying the script while it's running
timestamp=$(date +%Y%m%d%H%M%S)
mkdir -p scripts/temp
temp_script="scripts/temp/train_transfer_policy_temp_${timestamp}.sh"

if [ -f "scripts/train_transfer_policy.sh" ]; then
  cp "scripts/train_transfer_policy.sh" "$temp_script"
  echo "created copy : $temp_script"
else
  echo "ERROR: scripts/train_transfer_policy.sh not found"
  exit 1
fi

echo "total tasks ${num_tasks}, total gpus ${num_gpus}, polling allocation"

# start thread for every gpu
for gpu_idx in "${!gpu_device_ids[@]}"; do
  gpu_id=${gpu_device_ids[gpu_idx]}
  (
    echo "[GPU ${gpu_id}] subprocess started, task indexing"
    for (( i = gpu_idx; i < num_tasks; i += num_gpus )); do
      task=${tasks[i]}
      echo "[GPU ${gpu_id}] exec task: #$i → ${task}"
      bash "$temp_script" "${alg_name}" "${task}" "${info}" 0 "${gpu_id}"
    done
    echo "[GPU ${gpu_id}] tasks are done"
  ) &
done

wait
echo "all tasks done"
