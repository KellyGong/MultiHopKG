import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
# from transformers import AutoModel, AutoTokenizer


# model = "meta-llama/Llama-2-7b-chat-hf"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
"""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

tokenizer = LlamaTokenizer.from_pretrained('../llama')
model = LlamaForCausalLM.from_pretrained('../llama').to('cuda:1')

# pipeline = transformers.pipeline(
#     "text-generation",
#     tokenizer=tokenizer,
#     model=model,
#     torch_dtype=torch.float16,
#     device=1
# )

tokenizer.padding_side = "right"
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# sequences = pipeline(
#     "<s>[INST] <<SYS>>\n" + Prompt + "<</SYS>>\n\n" + 'Please select an answer, which is the capital of China, from the following candidate: [USA, Beijing, Guangzhou]. Please be concise.' + "[/INST]\n",
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200,
# )

# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")


def inference(tokenizer, model, input_string, device, sample_type='in'):
    system_prompt = f'<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n'
    # edit_prompt = []
    # labels = []
    # user_ids = []
    # for record in prompt:
    #     text = record['prompt']
    #     user_ids.append(record['id'])
    #     if sample_type == 'in':
    #         labels.append(text.split('\n\n')[-1].strip())
    #         text = '\n\n'.join(text.split('\n\n')[:-1])
    #     elif sample_type == 'out':
    #         labels.append(text)
    #     edit_prompt.append(system_prompt + text + ' [/INST]\n')
    input_string = system_prompt + input_string + ' [/INST]\n'
    inputs = tokenizer(input_string,
        padding=True,
        truncation=True,
        max_length=100,
        return_tensors='pt',)
    input_ids = inputs['input_ids'].to(device)
    # print(input_ids.shape)
    attention_mask = inputs['attention_mask'].to(device)
    # stopper = LogitsProcessorList([StopAfterSpaceIsGenerated(13, 2, device)])
    outputs = model.generate(input_ids=input_ids, 
                             attention_mask=attention_mask, 
                             eos_token_id=2, 
                             max_new_tokens=100,
                             remove_invalid_values=True,
    )
        # logits_processor=stopper)
    decoded = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:].detach(), skip_special_tokens=True)
    decoded = [i for i in decoded]
    return decoded

input_string = """Please select an answer, which is the capital of China, from the following candidate: [USA, Beijing, Guangzhou]. Please be concise."""

res = inference(tokenizer, model, input_string, 'cuda:1')

print(res)
