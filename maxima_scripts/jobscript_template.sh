#!/bin/bash
#SBATCH -n 1
#SBATCH -p general
#SBATCH --mem=5000
#SBATCH -t 40:00:00
#SBATCH --export=ALL
#SBATCH -J maxima_test
#SBATCH --output=maxima_sbatch.%A_%a.out
#SBATCH --error=maxima_sbatch.%A_%a.err
#SBATCH --mail-type=FAIL
