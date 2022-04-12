#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time-min=00:30:00
#SBATCH --time=10:00:00
#SBATCH --qos=flex
#SBATCH --requeue
#SBATCH --account=m4134
#SBATCH --constraint=haswell
#SBATCH --job-name=preproc-fourcastnet
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=begin,end,fail
#
# run EMAIL=$EMAIL sbatch ...
#
#SBATCH --mail-user=$EMAIL

# threads per worker = 2
export N_WORKERS=32
export N_THREADS_PER=2
export HDF5_USE_FILE_LOCKING='FALSE'
SCRIPT_DIR=$PWD
cd $SCRATCH

mkdir -p $SCRATCH/tmp
mkdir -p $SCRATCH/data/era5/fourcastnet

# stripe output dir, see https://docs.nersc.gov/performance/io/lustre/
# this is equivalent to: stripe_small data/era5/fourcastnet
lfs setstripe -c 8 $SCRATCH/data/era5/fourcastnet

module load python
source activate mjonet-preproc

# run dask from scratch, see https://docs.nersc.gov/analytics/dask/
python -u $SCRIPT_DIR/04-preproc-fourcastnet.py
