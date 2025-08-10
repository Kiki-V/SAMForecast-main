
model_name=SAMForecast
dropout=0.1
model_id_name=decomp
activation=silu
seq_len=96

for pred_len in 96 192 336 720
do
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 4 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512\
    --d_ff 512 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --itr 1 \
    --use_gpu True \
    --gpu 0 \
    --activation $activation\
    --layer_nums 1 \
    --k 1 \
    --expert_nums 3 \
    --wavelets_levels 3 4 5 \
    --dropout $dropout 
done
# >logs/test/'250208_'$model_name'_traffic_'$pred_len'_numlayer1_k1'.log
    # >logs/Traffic/final/$model_name'_'$pred_len'_[3,4,5]_numlayer'$numlayer'_k'$topk.log 
# python -u run_aaa.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_192 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 192 \
#   --e_layers 4 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1 \
#   --use_gpu True \
#   --gpu 0 \
#   --dropout $dropout >logs/$model_name'_'$model_id_name'_'$seq_len'_192_'$dropout'_'$activation'_transformer'.log 

# python -u run_aaa.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_336 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 336 \
#   --e_layers 4 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --d_model 512\
#   --d_ff 512 \
#   --batch_size 16 \
#   --learning_rate 0.001 \
#   --itr 1 \
#   --use_gpu True \
#   --gpu 0 \
#   --dropout $dropout >logs/$model_name'_'$model_id_name'_'$seq_len'_336_'$dropout'_'$activation'_transformer'.log 

# python -u run_aaa.py \
#   --is_training 1 \
#   --root_path ./dataset/traffic/ \
#   --data_path traffic.csv \
#   --model_id traffic_96_720 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 720 \
#   --e_layers 4 \
#   --enc_in 862 \
#   --dec_in 862 \
#   --c_out 862 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 16 \
#   --learning_rate 0.001\
#   --itr 1 \
#   --use_gpu True \
#   --gpu 0 \
#   --dropout $dropout >logs/$model_name'_'$model_id_name'_'$seq_len'_720_'$dropout'_'$activation'_transformer'.log 
