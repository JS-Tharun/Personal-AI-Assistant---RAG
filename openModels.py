from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)