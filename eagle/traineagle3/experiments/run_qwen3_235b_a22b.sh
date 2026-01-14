huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./Llama-3.3-70B-Instruct

deepspeed \
    --num_gpus 8 \
    main.py \
    --basepath ~/.cache/huggingface/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e/ \
    --trainpath sharegpt_expanded.jsonl \
    --deepspeed \
    --deepspeed_config ds_config_stage3.json \
    --configpath qwen3_235b_a22b.json \
    --template qwen
