CUDA_VISIBLE_DEVICES=0,1,2,3 \
IMAGE_MAX_TOKEN_NUM=2048 \
MASTER_PORT=29620 \
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift sft \
    --model /data/liyc/Code/med_vl/lingshu_7b \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --dataset /home/yuchong_li/KnowInject/data/rewrite_psv_train.json /home/yuchong_li/KnowInject/targeted_finetuning/lingshu7b_correction_only.json \
    --attn_impl flash_attn \
    --train_type full \
    --torch_dtype bfloat16 \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 4 \
    --save_strategy epoch \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir /path/to/output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --deepspeed zero2 \
    --loss_scale med

    # med loss scale: /home/yuchong_li/anaconda3/envs/swift28/lib/python3.11/site-packages/swift/loss_scale
    # model_type: gemma3_vision, qwen2_5_vl, qwen3_vl

