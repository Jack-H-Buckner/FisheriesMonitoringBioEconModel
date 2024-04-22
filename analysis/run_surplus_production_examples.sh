#!/bin/bash -l
# setting name of job
#SBATCH --job-name=VoI
#SBATCH --account=baskettgrp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
# setting home directory
#SBATCH -D /home/jhbuckne/SurplusProductionModel/FARM

# setting standard error output
#SBATCH -e /home/jhbuckne/SurplusProductionModel/FARM/slurm/stdoutput_%j.txt

# setting standard output
#SBATCH -o /home/jhbuckne/SurplusProductionModel/FARM/slurm/stdoutput_%j.txt

# setting medium priority
#SBATCH -p med2

# setting the max time
#SBATCH -t 6:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# send mail here
#SBATCH --mail-user=jhbuckner@ucdavis.edu

echo "Run: "
module load julia
julia -t 10 surplus_production_example_HCR.jl
julia -t 10 surplus_production_example_Policy.jl
julia -t 10 surplus_production_example_sim.jl
julia -t 10 surplus_production_example_VoI.jl
