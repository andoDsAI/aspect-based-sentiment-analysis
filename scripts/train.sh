export lr=3e-5
export coef=0.5
export seed=42
export epochs=100
export batch_size=16
echo "Training model with: "
echo "- Learning rate: ${lr}"
echo "- Seed: ${seed}"
echo "- Epochs: ${epochs}"
echo "- Batch size: ${batch_size}"
export MODEL_DIR=./trained_models
export DATA_DIR=./data
echo "- Save model to directory: ${MODEL_DIR}"
echo "- Load data from directory: ${DATA_DIR}"

python3 train.py --model_type phobert \
                  --model_dir $MODEL_DIR \
                  --data_dir $DATA_DIR \
                  --seed $seed \
                  --do_train \
				  --plot_result \
                  --eval_train \
                  --eval_dev \
                  --eval_test \
                  --epochs $epochs \
                  --batch_size $batch_size \
                  --aspect_coef $coef \
                  --threshold 0.5 \
                  --learning_rate $lr \
                  --max_seq_len 96 \
				  --embed_dim 256 \
				  --hidden_dim 256 \
				  --num_layers 12 \
				  --num_heads 4 \
                  --dropout_rate 0.2 \
                  --early_stopping 15
