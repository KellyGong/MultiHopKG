import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
# tokenizer = AutoTokenizer.from_pretrained("../llm/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("../llm/Baichuan2-7B-Chat", device_map={'': 'cuda:1'}, torch_dtype=torch.bfloat16, trust_remote_code=True)
# model.generation_config = GenerationConfig.from_pretrained("../llm/Baichuan2-7B-Chat")
# messages = []
# messages.append({"role": "user", "content": "Please select an answer, which is the capital of China, from the following candidate: [USA, Guangzhou, Beijing]. Please be concise."})
# response = model.chat(tokenizer, messages)
# print(response)


class LargeLM(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("../llm/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("../llm/Baichuan2-7B-Chat", device_map={'': 'cuda'}, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained("../llm/Baichuan2-7B-Chat")
    
    def prompt(self, e_s_i, e_i, q_i, next_r_i_list, next_e_i_list):
        next_action_dict = {"source_entity": e_s_i, "current_entity": e_i, "query_relation": q_i, "next_relation_entity_list": [(r, e) for r, e in zip(next_r_i_list, next_e_i_list)]}

        multi_hop_kg_reasoning_prompt = """The AI assistant performs multi-hop knowledge graph reasoning. The input is as the following format: {"source_entity": source_entity, "current_entity": current_entity, "query_relation": query_relation, "next_relation_entity_list": [(next_relation, next_entity)]}. The "source_entity" field denotes the source entity string, "current_entity" field denotes the current searched entity at current time step, while "query_relation" field expresses the query relation. The "next_relation_entity_list" fields denote the next search space, which is denotes as a list of (next relation, next entity). Please note that there exists a logical connections and order between the tasks. To assist with multi-hop knowledge graph reasoning, you can select one most possible (next relation, next entity) as the answer. The answer format should be {"next_relation": next_relation, "next_entity": next_entity}. """

        # transfer next_action_dict to string
        next_action_dict_str = json.dumps(next_action_dict)
        
        return multi_hop_kg_reasoning_prompt + "The question is " + next_action_dict_str

    def kg_next_entity(self, e_s, e, q, action_space_b, eid2entity, rid2relation):
        (next_r, next_e), action_mask = action_space_b
        messages = []
        for rollout_i in zip(e_s.tolist(), e.tolist(), q.tolist(), next_r.tolist(), next_e.tolist(), action_mask.tolist()):
            e_s_i, e_i, q_i, next_r_i, next_e_i, action_mask_i = rollout_i
            e_s_i = eid2entity[e_s_i]
            e_i = eid2entity[e_i]
            q_i = rid2relation[q_i]
            next_r_i_list = [rid2relation[r] for i, r in enumerate(next_r_i) if action_mask_i[i] == 1]
            next_e_i_list = [eid2entity[e] for i, e in enumerate(next_e_i) if action_mask_i[i] == 1]
            messages.append(self.prompt(e_s_i, e_i, q_i, next_r_i_list, next_e_i_list))
        response = self.generate(messages)
        pass

    def generate(self, messages, batch_size=64):
        input_messages = [{"role": "user", "content": message} for message in messages]
        responses = []
        # add tqdm for progress bar
        for i in tqdm(range(0, len(input_messages), batch_size)):
            response = self.model.chat(self.tokenizer, input_messages[i:i+batch_size])
            responses.append(response)
        return response
