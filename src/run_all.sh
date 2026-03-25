

nohup  bash -c "CUDA_VISIBLE_DEVICES=6 python LC_classification.py --llm_model BERT --tslm_model BERT --n_runs 1 --flare_process_type FE --id_process_type  uniq --ifP_tuning --ifLoRA " &


nohup  bash -c "CUDA_VISIBLE_DEVICES=1 python LC_classification.py --llm_model BERT --tslm_model BERT --n_runs 1 --flare_process_type FE --id_process_type  uniq --if_only_metavec --ifLoRA " &


nohup  bash -c "CUDA_VISIBLE_DEVICES=1 python LC_classification.py --llm_model BERT --tslm_model BERT --n_runs 1 --flare_process_type FA --id_process_type  same --ifP_tuning --ifLoRA " &


nohup  bash -c "CUDA_VISIBLE_DEVICES=2 python LC_classification.py --llm_model BERT --tslm_model BERT --n_runs 1 --flare_process_type PA --id_process_type  same --ifP_tuning --ifLoRA " &


nohup  bash -c "CUDA_VISIBLE_DEVICES=6 python LC_classification.py --llm_model BERT --tslm_model BERT --n_runs 1 --flare_process_type FA --id_process_type  same --if_only_metavec --ifLoRA " &


nohup  bash -c "CUDA_VISIBLE_DEVICES=6 python LC_classification.py --llm_model BERT --tslm_model BERT --n_runs 1 --flare_process_type PA --batch_size 64 --id_process_type  uniq --trend_period_extrac_type my --ifP_tuning --ifLoRA " &
