#!/bin/bash -e

#SBATCH --job-name=mpi_sum_compare
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

module purge
module load gimkl

# Build the executable if needed
make

# Run the MPI job
srun ./mpi_sum_compare


