import json, sys
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
system = {"role": "system", "content": "Answer in plain English."}
turns = [
    {"role": "user", "content": "The orchard key is called Cobalt Finch."},
    {"role": "assistant", "content": "I’ll remember Cobalt Finch for this conversation."},
    {"role": "user", "content": "My backup code is marigold."},
    {"role": "assistant", "content": "Got it."},
]
latest = {"role": "user", "content": "What is my backup code? 🍊"}
examples = {}
for omitted in range(3):
    messages = [system, *turns[omitted * 2:], latest]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, reasoning_effort="none"
    ) + "<|message_model|><|content_text|>"
    examples[prompt] = tokenizer.encode(prompt, add_special_tokens=False)
Path(sys.argv[2]).write_text(json.dumps(examples))
