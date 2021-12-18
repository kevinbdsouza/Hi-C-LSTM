#!/bin/bash
#SBATCH --account=def-maxwl
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=50000M        # memory per node
#SBATCH --time=00-02:00      # time (DD-HH:MM)
#SBATCH --output=output%N-%j_tf_.out  # %N for node name, %j for jobID

module load cuda cudnn hdf5 python/3.6.3
source /home/smaslova/pytorch/bin/activate

tensorboard --logdir=./tensorboard_logs/ --host 0.0.0.0 &
python -u train_model.py
