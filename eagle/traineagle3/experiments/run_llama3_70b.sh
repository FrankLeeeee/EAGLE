huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./Llama-3.3-70B-Instruct

deepspeed \
    --include localhost:0,1,2,3 \
    main.py \
    --basepath ./Llama-3.3-70B-Instruct \
    --trainpath sharegpt.jsonl \
    --deepspeed \
    --deepspeed_config ds_config_stage3.json \
    --configpath llama3_70b.json
