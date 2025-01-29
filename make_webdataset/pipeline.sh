cd /pscratch/qsh226_uksr/duolin/swiss-pos-contact/
#Step 1 preprocess PDB files into sperate npz files. Each npz file contains the index,seq and ca_pos
for i in {1..1084}  
do
data="/pscratch/qsh226_uksr/swiss-prot/folder-"${i}
outputfolder="/pscratch/qsh226_uksr/duolin/swiss-pos-contact/"
let index_base=$i*500-500
sbatch --export=data=$data,outputfolder=$outputfolder,index_base=$index_base, -J ${data} preprocess-mcc.sh

done

#Step 2
for j in {1..1084}
do
{

sbatch --export=j=$j, -J "tar-"${j} tar_files.sh
}
done


#Step 3
cd /pscratch/qsh226_uksr/duolin/swiss-pos-contact/

for j in {1..1084}
do
{

rm -rf folder-$j-*.npz
}
done
