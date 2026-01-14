huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./Llama-3.3-70B-Instruct

deepspeed \
    --num_gpus 8 \
    main.py \
    --basepath ~/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe/ \
    --trainpath sharegpt_expanded.jsonl \
    --deepspeed \
    --deepspeed_config ds_config_stage3.json \
    --configpath qwen3_30b_a3b.json \
    --template qwen

