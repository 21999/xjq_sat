# Examples:
# bash scripts/pretrain_policy.sh sat mix_pretrain 0322 0 0
# bash scripts/pretrain_policy.sh sat mix_pretrain "v3.1.0-sat_pretrain-all-100" 0 0


DEBUG=False
save_ckpt=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    eval_episodes=""
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    eval_episodes=""
    echo -e "\033[33mTrain mode\033[0m"
fi

# wandb tags
tags_json=$(printf '["%s","%s","%s"]' "$alg_name" "$task_name" "$addition_info")

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
export SDL_AUDIODRIVER=dummy

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

python pretrain.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            logging.project="sat_workspace" \
                            logging.tags="$tags_json" \
                            logging.name=${exp_name} \
                            ${eval_episodes} \
                            policy.num_inference_steps=5 \
                            task.dataset.max_train_episodes=90
