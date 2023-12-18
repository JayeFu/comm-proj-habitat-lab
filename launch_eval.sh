~/miniconda/condabin/conda init bash; source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate habitat

pip install -e habitat-lab
pip install -e habitat-baselines

EXP_CFG=$1
CKPT_NAME=$2
EVAL_CKPT_PATH_DIR=$3
TENSORBOARD_DIR=$4
LOG_FILE=$5

python -u -m habitat_baselines.run \
  --exp-config ${EXP_CFG} \
  --run-type eval \
  --ckpt_name ${CKPT_NAME} \
  habitat_baselines.eval_ckpt_path_dir=${EVAL_CKPT_PATH_DIR} \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.log_file=${LOG_FILE} \
  habitat_baselines.eval.video_option=['wandb'] \
  habitat_baselines.load_resume_state_config=False \
  "${@:6}"