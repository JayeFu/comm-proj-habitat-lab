~/miniconda/condabin/conda init bash; source ~/.bashrc; eval "$(conda shell.bash hook)"; conda activate habitat

pip install -e habitat-lab
pip install -e habitat-baselines

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
