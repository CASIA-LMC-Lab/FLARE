# FLARE


## Running 

**Installation**

Clone the repository locally:

```
git clone https://github.com/CASIA-LMC-Lab/FLARE.git
```

Install the required packages:

```
pip install -r requirements.txt
```


**dataset**

This project uses a dataset containing:
- kepler_flare_id_start_end.npy
- kepler_meta_feats3.npy
- kids.npy
- seqlen_512_horizon_48_stride_50_noPeriod.tar.gz 
- Kepler_npy.tar.gz

Please download it from:
- [Google Drive](https://drive.google.com/drive/folders/1vzjyoLW8UjiaUQAj9UegtCU9p_Jf6GWm?usp=drive_link) 


**run the training script**
```
python LC_classification.py \
    --llm_model BERT \
    --tslm_model BERT \
    --n_runs 3 \
    --flare_process_type FE \
    --id_process_type  uniq \
    --ifP_tuning \
    --ifLoRA 
```


