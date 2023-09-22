# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# torch.set_default_device('cuda')
# model = AutoModelForCausalLM.from_pretrained("../phi-1-5", trust_remote_code=True, torch_dtype="auto")
# tokenizer = AutoTokenizer.from_pretrained("../phi-1-5", trust_remote_code=True, torch_dtype="auto")
# inputs = tokenizer('''Please select an answer, which is the capital of China, from the following candidate: [USA, Beijing, Guangzhou]. Please be concise.''', return_tensors="pt", return_attention_mask=False)

# outputs = model.generate(**inputs, max_length=200)
# text = tokenizer.batch_decode(outputs)[0]
# print(text)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("../llm/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("../llm/Baichuan2-7B-Chat", device_map={'': 'cuda:1'}, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("../llm/Baichuan2-7B-Chat")
messages = []
messages.append({"role": "user", "content": "Please select an answer, which is the capital of China, from the following candidate: [USA, Guangzhou, Beijing]. Please be concise."})
response = model.chat(tokenizer, messages)
print(response)
