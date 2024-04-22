#!/bin/bash -l
# setting name of job
#SBATCH --job-name=tests

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
# setting home directory
#SBATCH -D /home/jhbuckne/SurplusProductionModel/FARM

# setting standard error output
#SBATCH -e /home/jhbuckne/SurplusProductionModel/FARM/slurm/stdoutput_%j.txt

# setting standard output
#SBATCH -o /home/jhbuckne/SurplusProductionModel/FARM/slurm/stdoutput_%j.txt

# setting medium priority
#SBATCH -p med2

# setting the max time
#SBATCH -t 96:00:00

# mail alerts at beginning and end of job
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# send mail here
#SBATCH --mail-user=jhbuckner@ucdavis.edu

# now we'll print out the contents of the R script to the standard output file
cat surplus_production_sims.jl
echo "ok now for the actual standard output"

module load julia
julia -t 60 surplus_production_sims_NCV_0.jl
