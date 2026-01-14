huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./Llama-3.3-70B-Instruct

deepspeed \
    --num_gpus 8 \
    main.py \
    --basepath ./Llama-3.3-70B-Instruct \
    --trainpath sharegpt_expanded.jsonl \
    --deepspeed \
    --deepspeed_config ds_config_stage3.json \
    --configpath llama3_70b.json
