export lr=3e-5
export coef=0.6
export seed=42
echo "${lr}"
export MODEL_DIR=Aspect-Model
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$coef"/"$seed
echo "${MODEL_DIR}"
python3 train.py --task ner \
                  --model_type xlmr \
                  --model_dir $MODEL_DIR \
                  --data_dir data/ \
                  --train_type train \
                  --val_type dev \
                  --test_type test \
                  --seed $seed \
                  --do_train \
                  --eval_train \
                  --eval_dev \
                  --eval_test \
                  --save_steps 69 \
                  --logging_steps 69 \
                  --num_train_epochs 500 \
                  --tuning_metric competition_score \
                  --gpu_id 0 \
                  --aspect_loss_coef $coef \
                  --threshold 0.5 \
                  --learning_rate $lr \
                  --max_seq_len 256 \
                  --train_batch_size 32 \
                  --eval_batch_size 64 \
                  --dropout_rate 0.4 \
                  --early_stopping 15 \
