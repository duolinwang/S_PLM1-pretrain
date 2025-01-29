#!/bin/bash

# V.Gazula 1/8/2019
 
#SBATCH -t 3:00:00   				#Time for the job to run 
#SBATCH -N 1 					#Number of nodes required
#SBATCH -n 1				#Number of cores needed for the job
#SBATCH --partition=normal  		#Name of the queue
#SBATCH --account=coa_qsh226_uksr		#Name of account to run under

# every tar file has 500 proteins
echo $j
tar -cf swiss-pro-pdbxyz-$j.tar folder-$j-*.npz
