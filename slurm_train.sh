#!/usr/bin/env bash
#SBATCH --partition=gpu # Hemera (HZDR) cluster
# necessary to set the account also to the queue name because otherwise access is not allowed at the moment
#SBATCH --time=23:59:00 
# Sets batch job's name
#SBATCH --job-name=train_phasegan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=5
##SBATCH --cpus-per-task=6
#SBATCH --mem=15900
##SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.ali@hzdr.de
##SBATCH --chdir=/home/aguilar72/Holography/demo_cwaveletflow
# notify the job 240 sec before the wall time ends
#SBATCH --signal=B:SIGALRM@240
# care the output folder
#SBATCH --error="train-%j.err"
#SBATCH --output="train-%j.out"

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)" # when running on Hemera (HZDR)
conda activate phasegan # nfphasing is a mamba (conda) environment
cd git/PhaseGAN

# care the output folder
python --version
python3 train.py --run_name 'test4' --num_epochs 1 --print_loss_freq_iter 20  --save_cycleplot_freq_iter 200 --load_path "bigdata/hplsim/scratch/aguila72/datasets"

echo "Waiting for job steps to complete..."
wait
echo "All job steps completed!"