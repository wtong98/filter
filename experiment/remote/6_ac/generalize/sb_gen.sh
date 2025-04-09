#!/bin/bash
#SBATCH -c 16
#SBATCH -t 1-00:00:00
#SBATCH -p seas_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-36%12
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=pehlevan_lab

source ../../../../../venv_haystack/bin/activate
python run.py ${SLURM_ARRAY_TASK_ID}

