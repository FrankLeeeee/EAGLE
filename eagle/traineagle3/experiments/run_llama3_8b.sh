deepspeed \
        --num_gpus 8 \
        main.py \
        --basepath ./Llama-3.1-8B-Instruct \
        --trainpath sharegpt.jsonl \
        --deepspeed \
        --deepspeed_config ds_config.json \
        --configpath config.json