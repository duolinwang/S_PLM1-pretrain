# S_PLM1-pretrain
Pretraining code for S-PLM-v1
Implementation of the pretraining pipeline for S-PLM-v1, including data processing and training loop.

## Data processing
1. Download protein structures in **PDB format** from [AlphaFold DB](https://alphafold.ebi.ac.uk/).
2. Run the preprocessing pipeline:
```bash
   bash pipeline.sh
```
## pre-training
Modify **data_path** in config files and run: 
```r
python train.py --config_path ./configs/config_16adapter.yaml
```
