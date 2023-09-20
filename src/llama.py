import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
# from transformers import AutoModel, AutoTokenizer


# model = "meta-llama/Llama-2-7b-chat-hf"

Prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

tokenizer = LlamaTokenizer.from_pretrained('../llama')
model = LlamaForCausalLM.from_pretrained('../llama', use_safetensors=True).to('cuda:1')

pipeline = transformers.pipeline(
    "text-generation",
    tokenizer=tokenizer,
    model=model,
    torch_dtype=torch.float16,
    device=1
)

sequences = pipeline(
    "<s>[INST] <<SYS>>" + Prompt + "<</SYS>>" + 'Please select an answer, which is the capital of China, from the following candidate: [USA, Beijing, Guangzhou]. Please be concise.' + "[/INST]",
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")