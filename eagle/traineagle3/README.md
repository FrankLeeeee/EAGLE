# Reproduction Steps

```bash
# create virtual env
uv venv .venv -p 3.9
source .venv/bin/activate

# install dependencies
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install datasets transformers==4.52.4 deepspeed importlib-metadata accelerate wandb

# prepare data
python prepare_sharegpt.py
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./Llama-3.1-8B-Instruct

# run training
deepspeed \
	--num_gpus 8 \
	main.py \
	--basepath ./Llama-3.1-8B-Instruct \
	--trainpath sharegpt.jsonl \
	--deepspeed \
	--deepspeed_config ds_config.json
```