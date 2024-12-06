#!/bin/bash
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -p kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-21
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_grads

source ../../../../../venv_haystack/bin/activate
python run.py ${SLURM_ARRAY_TASK_ID}

