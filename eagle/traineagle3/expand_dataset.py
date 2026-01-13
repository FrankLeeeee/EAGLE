from datasets import load_dataset
from transformers import AutoTokenizer
import json
from tqdm import tqdm



def main():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    dataset = load_dataset("json", data_files="sharegpt.jsonl")

    with open("sharedgpt_expanded.jsonl", "w") as f:
        for item in tqdm(dataset['train']):
            conversations = item["conversations"]
            last_message = conversations[-1]

            if last_message["role"] != "assistant":
                continue
            else:
                response = last_message["content"]
                response_ids = tokenizer.encode(response, add_special_tokens=False)
                response_ids_len = len(response_ids)
                if response_ids_len < 4096:
                    num_tokens_to_add = 4096 - response_ids_len
                    response = response + ' ' * num_tokens_to_add
                    last_message["content"] = response
                f.write(json.dumps(item) + "\n")    

if __name__ == "__main__":
    main()