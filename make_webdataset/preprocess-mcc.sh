#!/bin/bash

# V.Gazula 1/8/2019

#SBATCH -t 24:00:00   				#Time for the job to run 
#####SBATCH --job-name=prep		   	#Name of the job
#SBATCH -N 1 					#Number of nodes required
#SBATCH -n 1				#Number of cores needed for the job
#SBATCH --partition=normal  		#Name of the queue
#SBATCH --account=coa_qsh226_uksr		#Name of account to run under

module load Miniconda3 
source activate /project/qsh226_uksr/qsh226/simclr-cpu

#for file in *pdb.gz
#do
#{
#	gunzip -k $file
#}
#done
echo ${data}
echo ${outputfolder}
echo ${index_base}

python3 preprocess_pdbxyz_contact.py -data ${data} -max_len 512 -outputfolder ${outputfolder} -index_base ${index_base}
