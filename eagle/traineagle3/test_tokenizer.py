from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": "Large language model is a type of machine learning model that is trained to understand and generate human language."}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
)

print(text)

