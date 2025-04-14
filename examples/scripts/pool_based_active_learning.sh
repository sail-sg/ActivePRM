conda activate o1

base_dir=./out/models
model=Qwen/Qwen2.5-Math-7B-Instruct
model_id=$(echo "$model" | cut -d'/' -f2)

problem_type=single_label_classification
label_type=hard_labels

num_ensemble=32 # 8, 1
p_threshold=5
std_threshold=-1
lr=1e-5
batch_size=64
dataset=$1

if [[ $num_ensemble == 1 ]]; then
  learning_probability=1.0
  regularization_lambda=0.0
else
  learning_probability=1.0
  regularization_lambda=0.5
fi
freeze_backbone=False
# echo $annotate_model $num_ensemble $learning_probability $freeze_backbone

echo num_ensemble $num_ensemble
echo learning_probability $learning_probability
echo pred_threshold $p_threshold
echo std_threshold $std_threshold
echo lr $lr
echo batch_size $batch_size
echo dataset $dataset

exp_name=pool_based_active_learning
model_id=$exp_name
output_dir=$base_dir/${exp_name}
mkdir -p $output_dir

enable_wandb=True
if [[ $enable_wandb == 'True' ]]; then
  wandb online
  export WANDB_PROJECT=active_prm
  export WANDB_RESUME='allow'
  export WANDB_RUN_ID=$exp_name
  report_to='wandb'
else
  report_to='none'
fi

# training hps
num_train_epochs=1
per_device_train_batch_size=8
gradient_accumulation_steps=$((batch_size / (per_device_train_batch_size * num_gpus)))

accelerate launch py_scripts/pool_based_active_learning.py \
  --deepspeed ds_config.json \
  --model_name_or_path $model \
  --dataset_name ${dataset} \
  --learning_rate $lr \
  --num_train_epochs $num_train_epochs \
  --per_device_train_batch_size $per_device_train_batch_size \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --gradient_checkpointing \
  --logging_steps 5 \
  --logging_dir $output_dir/logs \
  --save_total_limit 1 \
  --save_strategy 'steps' \
  --save_steps 100 \
  --eval_strategy 'steps' \
  --eval_steps 500 \
  --output_dir $output_dir \
  --report_to=$report_to \
  --run_name $exp_name \
  --max_seq_length 2048 \
  --bf16=True \
  --torch_dtype auto \
  --num_ensemble $num_ensemble \
  --label_type $label_type \
  --problem_type $problem_type \
  --learning_probability $learning_probability \
  --regularization_lambda $regularization_lambda \
  --rr_token '<extra_0>' \
  --active_learning_pred_threshold $p_threshold \
  --active_learning_std_threshold $std_threshold \
  --lr_scheduler_type 'linear' \
  --warmup_steps 500 \
  2>&1 | tee $output_dir/log.txt

python -m online_prm.eval.processbench prm ${output_dir}

accelerate launch -m active_prm.eval.PRMBench \
  --model ensemble_prm \
  --model_args pretrained=${output_dir} \
  --task_name prmtest_classified \
  --verbosity INFO \
  --output_path ./out/bench/prmbench/${model_id}/results.jsonl

python -m online_prm.eval.PRMBench.vis_res ./out/bench/prmbench/${exp_name}/results.jsonl
