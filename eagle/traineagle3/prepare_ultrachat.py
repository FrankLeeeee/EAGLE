      
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/ultrachat_200k")
import json

with open("./ultrachat_200k.jsonl", "w") as f:
    for item in ds["train_sft"]:
        conversations = item['messages']
        new_conversations = []

        for message in conversations:
            role = message['role']
            content = message['content']
            assert role in ["user", "assistant"]
            new_conversations.append({
                "role": role,
                "content": content
            })

        row = {
            "id": item["prompt_id"],
            "conversations": new_conversations
        }

        f.write(json.dumps(row) + "\n")#

    