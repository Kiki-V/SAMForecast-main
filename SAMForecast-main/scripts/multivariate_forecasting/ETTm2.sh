
model_name=SAMForecast
dropout=0.1
model_id_name=decomp
activation=silu
seq_len=96

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_96 \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --pred_len $pred_len \
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
    --expert_nums 3 \
    --wavelets_levels 3 4 5 \
    --gpu 0 
done
