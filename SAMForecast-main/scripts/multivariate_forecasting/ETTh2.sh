# export CUDA_VISIBLE_DEVICES=1

model_name=ddd_final_test
dropout=0.1
model_id_name=decomp
activation=silu
seq_len=96

python -u run_aaa.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 192 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --use_gpu True \
  --activation $activation\
  --layer_nums 2 \
  --k 2 \
  --expert_nums 2 \
  --wavelets_levels 3 4 \
  --gpu 0 >logs/seq192/'250524_'$model_name'_ETTh2_96'.log

python -u run_aaa.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 192 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --use_gpu True \
  --activation $activation\
  --layer_nums 2 \
  --k 2 \
  --expert_nums 2 \
  --wavelets_levels 3 4 \
  --gpu 0 >logs/seq192/'250524_'$model_name'_ETTh2_192'.log

  
python -u run_aaa.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 192 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --use_gpu True \
  --activation $activation\
  --layer_nums 1 \
  --k 2 \
  --expert_nums 2 \
  --wavelets_levels 3 4 \
  --gpu 0 >logs/seq192/'250524_'$model_name'_ETTh2_336'.log

  
python -u run_aaa.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 192 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --use_gpu True \
  --activation $activation\
  --layer_nums 1 \
  --k 3 \
  --expert_nums 3 \
  --wavelets_levels 2 3 4 \
  --gpu 0 >logs/seq192/'250524_'$model_name'_ETTh2_720'.log
#   python -u run_aaa.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_96 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --use_gpu True \
#   --activation $activation\
#   --layer_nums 1 \
#   --k 2 \
#   --expert_nums 2 \
#   --wavelets_levels 3 4 \
#   --gpu 0 >logs/test/'250208_'$model_name'_ETTh2_336_numlayer1_k2_[3,4]'.log

  # python -u run_aaa.py \
  # --is_training 1 \
  # --root_path ./dataset/ETT-small/ \
  # --data_path ETTh2.csv \
  # --model_id ETTh2_96_96 \
  # --model $model_name \
  # --data ETTh2 \
  # --features M \
  # --seq_len 96 \
  # --pred_len 720 \
  # --e_layers 2 \
  # --enc_in 7 \
  # --dec_in 7 \
  # --c_out 7 \
  # --des 'Exp' \
  # --d_model 128 \
  # --d_ff 128 \
  # --itr 1 \
  # --use_gpu True \
  # --activation $activation\
  # --layer_nums 1 \
  # --k 4 \
  # --expert_nums 4 \
  # --wavelets_levels 2 3 4 5 \
  # --gpu 0 >logs/test/'250208_'$model_name'_ETTh2_720_numlayer1_k4_[2,3,4,5]'.log

# >logs/ETT/$model_name'_720_ETTh2_decomp_staNorm_[2,3,4,5]_numlayer2_k2'.log 
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_192 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --use_gpu True \
#   --gpu 0 \

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_336 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --use_gpu True \
#   --gpu 0 \

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_720 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --use_gpu True \
#   --gpu 0 \