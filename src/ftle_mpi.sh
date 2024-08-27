#!/bin/bash
#SBATCH --qos=preemptable
#SBATCH --nodes=6
#SBATCH --time=03:00:00
#SBATCH --ntasks=180
#SBATCH --job-name=ftle_mpi
#SBATCH --constraint="edr"
#SBATCH --output=/rc_scratch/elst4602/LCS_project/ftle_script_msgs.out

module purge

module load anaconda
conda activate /projects/elst4602/software/anaconda/envs/mpipyenv
module load intel/2022.1.2 
module load impi

export SLURM_EXPORT_ENV=ALL

mpirun -genv I_MPI_FABRICS shm:ofi -np $SLURM_NTASKS /projects/elst4602/software/anaconda/envs/mpipyenv/bin/python3 /home/elst4602/src_code/FTLE_mpi.py
