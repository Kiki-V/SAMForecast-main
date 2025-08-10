

model_name=SAMForecast
dropout=0.1
model_id_name=decomp
activation=silu
seq_len=96

python -u run.py \
      --is_training 1 \
      --root_path ./dataset/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id Exchange_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 192 \
      --pred_len 96 \
      --e_layers 2 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --itr 1 \
      --use_gpu False \
      --activation $activation\
      --layer_nums 3 \
      --k 2 \
      --expert_nums 3 \
      --wavelets_levels 3 4 5 \
      --gpu 0  

python -u run.py \
      --is_training 1 \
      --root_path ./dataset/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id Exchange_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 192 \
      --pred_len 192 \
      --e_layers 2 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --itr 1 \
      --use_gpu False \
      --activation $activation\
      --layer_nums 3 \
      --k 2 \
      --expert_nums 3 \
      --wavelets_levels 3 4 5 \
      --gpu 0   
      
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id Exchange_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 192 \
      --pred_len 336 \
      --e_layers 2 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --itr 1 \
      --use_gpu True \
      --activation $activation\
      --layer_nums 2 \
      --k 3 \
      --expert_nums 3 \
      --wavelets_levels 3 4 5 \
      --gpu 0  
      
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/exchange_rate/ \
      --data_path exchange_rate.csv \
      --model_id Exchange_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 192 \
      --pred_len 720 \
      --e_layers 2 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --d_model 128 \
      --d_ff 128 \
      --itr 1 \
      --use_gpu True \
      --activation $activation\
      --layer_nums 2 \
      --k 3 \
      --expert_nums 3 \
      --wavelets_levels 3 4 5 \
      --gpu 0  
