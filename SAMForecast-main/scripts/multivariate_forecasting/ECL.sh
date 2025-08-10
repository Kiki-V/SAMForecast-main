model_name=SAMForecast
dropout=0.1
model_id_name=decomp
activation=silu
seq_len=96

for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 512 \
      --batch_size 16 \
      --learning_rate 0.0005 \
      --itr 1 \
      --use_gpu True \
      --activation $activation\
      --layer_nums 2 \
      --k 1 \
      --expert_nums 3 \
      --wavelets_levels 3 4 5 \
      --gpu 0  
done
