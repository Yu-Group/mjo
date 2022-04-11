#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --qos=regular
#SBATCH --account=m4134
#SBATCH --constraint=knl
#SBATCH --job-name=preproc-fourcastnet
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=$EMAIL

module load python
conda activate mjonet-preproc

srun python 04-preproc-fourcastnet.py
