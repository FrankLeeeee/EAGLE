import argparse
import deepspeed
import re

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/llama31chat/8B/')
parser.add_argument('--trainpath', type=str,
                    default="/home/lyh/code/nlp/developing/vllmbase/vllm/gedata/l318b.jsonl")
# parser.add_argument('--testpath', type=str,
#                     default="/home/lyh/code/nlp/developing/vllmbase/vllm/gedata/0318.json")
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--savedir', type=str, default='0')
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
parser.add_argument("--template", type=str, choices=["llama", "qwen"], default="llama")

parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
import json
import re
import time

deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)
train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": 40,
    "num_workers": 2,
    "max_len": 4096,
    "config_path": args.configpath,
}

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from cnets import padding

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate.utils import set_seed

set_seed(0)
from cnets import Model
from configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup



def build_dataset_rank(
        tokenizer, datapath, template
):

    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = 64

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples['conversations'][i]
            if not source:
                continue
            if source[0]["role"] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for j, sentence in enumerate(source):
                role = sentence["role"]
                assert role == convroles[j % 2], f"{i}"
                # if sentence["from"]=="gpt":
                #     sentence["value"]=" "+sentence["value"]
                messages.append(
                    {"role": role, "content": sentence["content"]}
                )
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=4096,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)
            # print(i)

            if template == "llama":
                assistant_message_separator = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                end_of_turn_token = "<|eot_id|>"
            else:
                assistant_message_separator = "<|im_start|>assistant\n"
                end_of_turn_token = "<|im_end|>\n"

            assistant_pattern = (
                        re.escape(assistant_message_separator)
                        + r"([\s\S]*?(?:"
                        + re.escape(end_of_turn_token)
                        + "|$))"
                    )

            # get input_ids
            loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
            for match in re.finditer(assistant_pattern, conversation, re.DOTALL):
                content_start_char = match.start(1)
                content_end_char = match.end(1)

                # --- Core Alternative Operation: Calculate Token Index Based on Prefix String Length ---
                # Encode the text "assistant start", the length of which is the position of the starting token.
                prefix_ids = tokenizer.encode(
                    conversation[:content_start_char], add_special_tokens=False
                )
                # Encodes the text "assistant end", the length of which is the position of the end token.
                full_ids = tokenizer.encode(
                    conversation[:content_end_char], add_special_tokens=False
                )

                start_token_idx = len(prefix_ids)
                end_token_idx = len(full_ids)

                # Handling out-of-bounds errors caused by truncation
                actual_start = min(start_token_idx, len(input_ids))
                actual_end = min(end_token_idx, len(input_ids))

                if actual_start < actual_end:
                    loss_mask[actual_start:actual_end] = 1


            # new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(torch.ones_like(loss_mask)[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
    )


    ds1.set_format(type="torch")
    return ds1


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


config = EConfig.from_pretrained(train_config["config_path"])

from contextlib import nullcontext

print(f"Running with zero stage {ds_config['zero_optimization']['stage']}")
is_stage_3 = ds_config["zero_optimization"]["stage"] == 3
with nullcontext() if not is_stage_3 else deepspeed.zero.Init():
    model = Model(config, path=args.basepath, load_emb=True, load_head=True, use_pretrained_weights=not is_stage_3)
model.scandata(args.trainpath, args.basepath, args.template)

tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset_rank(tokenizer, args.trainpath, args.template)
print(f"Finished building data set. ")
# testdataset = build_dataset_rank(tokenizer, args.testpath)



criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]

model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     )
model = model.cuda()

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()

# if global_rank == 0:
#     import wandb
#     wandb.init(project="specforge-debug", name="official-eagle3-deepspeed-llama3", config=ds_config)

os.makedirs(args.savedir, exist_ok=True)

# sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
# test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4, pin_memory=True,
#                          collate_fn=DataCollatorWithPadding())

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, num_workers=4,
                          pin_memory=True,
                          collate_fn=DataCollatorWithPadding())


def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1


checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
# if checkpoint_path:
#     print(f"load from {checkpoint_path}")
#     model_engine.load_checkpoint(checkpoint_path)



for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch+1)
    print(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]
    global_step = 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        global_step += 1

        if global_step == 11:
            torch.cuda.synchronize()
            start_time = time.time()

        model.zero_grad()

        plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
                                               attention_mask=data["attention_mask"].to(rank),
                                               loss_mask=data["loss_mask"],
                                               )

        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss
        model_engine.backward(loss)
        model_engine.step()

        if global_step == 50:
            torch.cuda.synchronize()
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds, average time per step: {(end_time - start_time) / 40} seconds")
            exit()

        # if global_rank == 0:
        #     logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
        #     for i in range(len(plosses)):
        #         logdict[f"train/ploss_{i}"] = plosses[i].item()
        #     for i in range(len(acces)):
        #         logdict[f"train/acc_{i}"] = acces[i]
        #     wandb.log(logdict)
        # epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        # epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]


    # for i in range(len(epoch_acces)):
    #     acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
    #     deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
    #     acc_i = acc_i.item()
    #     if global_rank == 0:
    #         wandb.log({f"train/epochacc_{i}": acc_i})
    #         print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    # for i in range(len(epoch_plosses)):
    #     loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
    #     deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
    #     loss_i = loss_i.item()
    #     if global_rank == 0:
    #         wandb.log({f"train/epochploss_{i}": loss_i})
    #         print(f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")

    # epoch_acces = [[] for _ in range(model.length)]
    # epoch_plosses = [[] for _ in range(model.length)]

    # for batch_idx, data in enumerate(tqdm(test_loader)):
    #     with torch.no_grad():
    #         plosses, vlosses, acces = model_engine(input_ids=data["input_ids"].to(rank),
    #                                                attention_mask=data["attention_mask"].to(rank),
    #                                                loss_mask=data["loss_mask"],
    #                                                )
    #         epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
    #         epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    # for i in range(len(epoch_acces)):
    #     acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
    #     deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
    #     acc_i = acc_i.item()
    #     if global_rank == 0:
    #         wandb.log({f"test/epochacc_{i}": acc_i})
    #         print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}")

    # for i in range(len(epoch_plosses)):
    #     loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
    #     deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
    #     loss_i = loss_i.item()
    #     if global_rank == 0:
    #         wandb.log({f"test/epochploss_{i}": loss_i})
    #         print(f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}")


    # model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    # if epoch % 10 == 0:
    #     deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_{epoch}")
