# S_PLM1-pretrain
Pretraining code for S-PLM-v1
Implementation of the pretraining pipeline for S-PLM-v1, including data processing and training loop.

## Data processing
1. Download protein structures (Swiss-Prot (PDB files)) in **PDB format** from [AlphaFold DB](https://alphafold.ebi.ac.uk/download).
2. Run the preprocessing pipeline in the folder make_webdataset:
```bash
   bash pipeline.sh
```
## Pre-training for S-PLM-v1
Modify **data_path** in config files and run: 
```r
python train.py --config_path ./configs/config_16adapter.yaml
```
