#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=compute

# The default run (wall-clock) time is 1 minute
#SBATCH --time=15:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# The default number of CPUs per task is 1, however CPUs are always allocated per 2, so for a single task you should use 2
#SBATCH --cpus-per-task=1

# The default memory per node is 1024 megabytes (1GB)
#SBATCH --mem-per-cpu=32G

# Set mail type to 'END' to receive a mail when the job finishes (with usage statistics)
#SBATCH --mail-type=END

# Your job commands go below here

# Uncomment these lines when your job requires this software

module load slurm
module load 2022r1
module load compute
# module load python/3.8.12-p6aunbm
module load py-numpy


module load py-pip
python -m pip install --user wheel
python -m pip install --user pandas
python -m pip install --user scipy
python -m pip install --user scikit-learn==0.24.2
python -m pip install --user pickle5
python -m pip install --user openpyxl
python -m pip install --user seaborn


srun Final_model_1M_randomization.py
