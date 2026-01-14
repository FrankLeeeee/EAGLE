huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./Llama-3.1-8B-Instruct

deepspeed \
        --num_gpus 8 \
        main.py \
        --basepath ./Llama-3.1-8B-Instruct \
        --trainpath sharegpt_expanded.jsonl \
        --deepspeed \
        --deepspeed_config ds_config.json \
        --configpath config.json