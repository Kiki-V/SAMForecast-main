
model_name=SAMForecast
dropout=0.1
model_id_name=decomp
activation=silu
seq_len=96

for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id weather_96_192 \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --d_model 512\
      --d_ff 512\
      --itr 1 \
      --layer_nums 3 \
      --k 2 \
      --use_gpu False \
      --expert_nums 3 \
      --wavelets_levels 3 4 5 \
      --gpu 0 
done