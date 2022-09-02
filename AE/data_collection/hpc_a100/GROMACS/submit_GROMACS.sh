#!/bin/bash
#SBATCH --job-name=GROMACS_V100_DATA_1
#SBATCH --output=%x.%j.o
#SBATCH --error=%x.%j.e
#SBATCH --partition matador
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=40

./launch
