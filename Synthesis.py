import re
from langchain.output_parsers.json import parse_json_markdown
import copy
from zhipuai import ZhipuAI
from transformers import AutoTokenizer, AutoModel
from fuzzywuzzy import fuzz


def try_fix(reponse_text: str, key: str, extract_key: str):
    if isinstance(key, str):
        similarity = fuzz.ratio(key, extract_key)
        replace_key = key
    elif isinstance(key, list):
        similarity = 0
        replace_key = ""
        for key_ in key:
            if fuzz.ratio(key_, extract_key) > similarity:
                similarity = fuzz.ratio(key_, extract_key)
                replace_key = key_
    if similarity >= 0.50:
        if extract_key in reponse_text:
            reponse_text = reponse_text.replace(extract_key, replace_key)
    return reponse_text


class ChatModel:
    def chat(self, input_text, history, **kwargs):
        pass

    def __call__(self, input_text, history, **kwargs):
        return self.chat(input_text, history, **kwargs)


class LocalGLMChatModel(ChatModel):
    def __init__(self, model_name_or_path,):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

    def chat(self, input_text="", history=None, do_sample=True, **kwargs):
        if not do_sample:
            response, _ = self.model.chat(self.tokenizer, input_text, history=history, top_k=1, max_new_tokens=4096, max_length=None)
        else:
            response, _ = self.model.chat(self.tokenizer, input_text, history=history, do_sample=do_sample, max_new_tokens=4096, max_length=None)
        return response

    def __call__(self, input_text="", history=None, do_sample=True, **kwargs):
        return self.chat(input_text=input_text, history=history, do_sample=do_sample,**kwargs)


class GLMChatModel(ChatModel):
    def __init__(self, key, model_type="glm-4", do_sample=True):
        self.client = ZhipuAI(api_key=key)
        self.model_type = model_type
        self.do_sample = do_sample

    def chat(self, input_text="", history=None, **kwargs):
        if history is not None:
            messages = copy.deepcopy(history)
        else:
            messages = []
        messages.append({"role": "user", "content": input_text},)
        response = self.client.chat.completions.create(
            model=self.model_type,  # 填写需要调用的模型名称
            messages=messages,
            do_sample=self.do_sample,
        )
        messages.append({"role":"assistant", "content":response.choices[0].message.content})
        return response.choices[0].message.content, messages

    def __call__(self,  input_text="", history=None, **kwargs):
        return self.chat(input_text=input_text, history=history, **kwargs)


class OutputParserError(Exception):
    """Exception raised when parsing output from a command fails."""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "Failed to parse output of the model:%s\n " % self.message


class OutputParser(object):
    def __init__(self, keys, force_parser=False, max_length=4096):
        self.keys = keys
        self.force_parser = force_parser

    def parser(self, text):
        text = text.replace("'", "\"")
        try:
            result = parse_json_markdown(text)
            for key in result:
                if isinstance(result[key], list):
                    result[key] = result[key][0]
            keys_missing = False
            if self.force_parser:
                for key in self.keys:
                    if result.get(key, False):
                        pass
                    else:
                        keys_missing = True
                        break
            else:
                for key in self.keys:
                    if result.get(key, False):
                        pass
                    else:
                        result[key] = 0
            if not keys_missing:
                return {"success":True, "result":result, "cleaned_response":text}
            else:
                return {"success":False, "result":result, "cleaned_response":text}
        except:
            raise OutputParserError(text)

    def __call__(self, text=""):
        return self.parser(text)


class Agent(object):
    def __init__(self, chat_model: ChatModel, parser: OutputParser,
                 core_function_prompt="",
                 format_function_prompt="",
                 enhance_function_prompt="",
                 max_retry=5,
                 **kwargs):
        self.chat_model = chat_model
        self.core_function_prompt = core_function_prompt
        self.format_function_prompt = format_function_prompt
        self.enhance_function_prompt = enhance_function_prompt
        self.PROMPT_TEMPLATE = "{}\n{}\n{}"
        self.history = []
        self.max_retry = max_retry
        self.parser = parser

    def build_input_text(self, input_text, use_core_function_prompt=True, **kwargs):
        if use_core_function_prompt:
            return self.PROMPT_TEMPLATE.format(
                self.core_function_prompt.format(input_text),
                self.format_function_prompt,
                self.enhance_function_prompt, input_text)
        else:
            return self.PROMPT_TEMPLATE.format(
                input_text,
                self.format_function_prompt,
                self.enhance_function_prompt)

    def response_without_memory(self, input_text, **kwargs, ):
        retry = 0
        while retry < self.max_retry:
            try:
                self.clear_history()
                response_text, historys = self.chat_model(self.build_input_text(input_text), self.history, **kwargs)
                parser_result = self.parser(response_text)
                self.history = historys
                if parser_result["success"]:
                    return parser_result
            except OutputParserError:
                pass
            retry += 1
        return {"success":False,}


    def response_with_memory(self, input_text, **kwargs):
        retry = 0
        while retry < self.max_retry:
            try:
                response_text, historys = self.chat_model(self.build_input_text(input_text, use_core_function_prompt=False), self.history, **kwargs)
                parser_result = self.parser(response_text)
                self.history = historys
                if parser_result["success"]:
                    return parser_result
            except OutputParserError:
                pass
            retry += 1
        return {"success":False,}

    def response(self, input_text, use_memory=False, **kwargs):
        if use_memory:
            return self.response_with_memory(input_text, **kwargs)
        else:
            return self.response_without_memory(input_text, **kwargs)

    def add_history(self, history):
        self.history.append(history)

    def clear_history(self):
        self.history = []



class TwoGLMAgents(object):
    def __init__(self, key, model_type="glm-4"):
        chat_model_1 = GLMChatModel(key, model_type, do_sample=True)
        chat_model_2 = GLMChatModel(key, model_type, do_sample=False)
        parser_1 = OutputParser(keys=["sentence"], force_parser=True)
        parser_2 = OutputParser(keys=["location", "person", ], force_parser=False)
        self.generator = Agent(chat_model_1, parser_1,
                               core_function_prompt="写一句话，需要包含{}。",
                               format_function_prompt="\n按照以下json格式输出:\n\n```json\n{\"sentence\":按要求写的句子}\n```")

        format_function_prompt = '''\n按照以下json格式输出:\n
```json\n{"person":句子中包含的人物, "location":句子包含的地点, }\n```
'''
        self.extractor = Agent(chat_model_2, parser_2,
                               core_function_prompt="按照要求回答，这句话中包含的人物、地点。这是需要分析的句子：\n{}",
                               format_function_prompt=format_function_prompt)


    def action(self, input_text, check_keys, max_retry=10):
        retry = 0
        generate_text = ""
        extract_results = {}
        while retry < max_retry:
            response = self.generator.response(input_text)
            if response["success"]:
                generate_text = response["result"]["sentence"]
            else:
                retry += 1
                continue
            response = self.extractor.response(generate_text)
            if response["success"]:
                extract_results = response["result"]
            else:
                retry += 1
                continue
            action_can_finish = True
            for k, v in check_keys.items():
                if extract_results[k] not in v:
                    action_can_finish = False
                    break
            if not action_can_finish:
                retry += 1
            else:
                return True, generate_text, extract_results, retry
        return False, generate_text, extract_results, retry

    def action_0223(self, input_text, check_keys, max_retry=10):
        retry = 0
        generate_text = ""
        extract_results = {}
        while retry < max_retry:
            response = self.generator.response(input_text)
            if response["success"]:
                generate_text = response["result"]["sentence"]
            else:
                retry += 1
                continue
            response = self.extractor.response(generate_text)
            if response["success"]:
                extract_results = response["result"]
            else:
                retry += 1
                continue
            action_can_finish = True
            for k, v in check_keys.items():
                if extract_results[k] not in v:
                    action_can_finish = False
                    break

            ### try fix up
            if not action_can_finish:
                generate_text_fixup = copy.deepcopy(generate_text)
                for k, v in check_keys.items():
                    generate_text_fixup = try_fix(generate_text_fixup, v, extract_results[k])
                response = self.extractor.response(generate_text_fixup)
                generate_text = copy.deepcopy(generate_text_fixup)
                if response["success"]:
                    extract_results = response["result"]
                else:
                    retry += 1
                    continue
                action_can_finish = True
                for k, v in check_keys.items():
                    if extract_results[k] not in v:
                        action_can_finish = False
                        break

            if not action_can_finish:
                retry += 1
            else:
                return True, generate_text, extract_results, retry
        return False, generate_text, extract_results, retry

    def action_0328(self, input_text, check_keys, max_retry=10):
        retry = 0
        generate_text = ""
        extract_results = {}
        while retry < max_retry:
            if retry == 0:
                response = self.generator.response(input_text,)
            else:
                response = self.generator.response("不符合要求，重写", use_memory=True)
            if response["success"]:
                generate_text = response["result"]["sentence"]
            else:
                retry += 1
                continue
            response = self.extractor.response(generate_text)
            if response["success"]:
                extract_results = response["result"]
            else:
                retry += 1
                continue
            action_can_finish = True
            for k, v in check_keys.items():
                if extract_results[k] not in v:
                    action_can_finish = False
                    break
            if not action_can_finish:
                retry += 1
            else:
                return True, generate_text, extract_results, retry
        return False, generate_text, extract_results, retry


    def action_ext_0223(self, input_text, check_keys,):
        extract_results = {}
        response = self.extractor.response(input_text)
        if response["success"]:
            extract_results = response["result"]
        else:
            return False, extract_results
        action_can_finish = True
        for k, v in check_keys.items():
            if extract_results[k] not in v:
                action_can_finish = False
                break

        if not action_can_finish:
            return False, extract_results
        else:
            return True, extract_results


def encode(Coders, Coder_names, bit_streams, return_single=True, alg="flc"):
    entitys = []
    concept_paths = []
    bit_index = 0
    for Coder_name, Coder in zip(Coder_names, Coders):
        if Coder_name == "T":
            if alg == "flc":
                entitys_tmp, concept_path_tmp, _ = Coder.encode(bit_streams[bit_index:])
            elif alg == "hc":
                entitys_tmp, concept_path_tmp, _ = Coder.encode_hc(bit_streams[bit_index:])
            elif alg == "ac":
                entitys_tmp, concept_path_tmp, _ = Coder.encode_ac(bit_streams[bit_index:])
            bit_num_used = int(np.floor(np.log2(len(entitys_tmp))))
            entitys.append([np.random.choice(entitys_tmp)])
            concept_paths.append(concept_path_tmp)
            bit_index += bit_num_used
        else:
            if alg == "flc":
                entitys_tmp, concept_path_tmp, bit_num_used = Coder.encode(bit_streams[bit_index:])
            elif alg == "hc":
                entitys_tmp, concept_path_tmp, bit_num_used = Coder.encode_hc(bit_streams[bit_index:])
            elif alg == "ac":
                entitys_tmp, concept_path_tmp, bit_num_used = Coder.encode_ac(bit_streams[bit_index:])
            entitys_tmp = sorted(entitys_tmp, key=lambda x: len(x), )
            bit_index += bit_num_used
            entitys_better = [entity for entity in entitys_tmp if not bool(re.search('[a-z]', entity))]
            if return_single:
                if len(entitys_better):
                    entitys.append(np.random.choice(entitys_better))
                else:
                    entitys.append(np.random.choice(entitys_tmp))
            else:
                if len(entitys_better):
                    entitys.append(entitys_better)
                else:
                    entitys.append(entitys_tmp)
            concept_paths.append(concept_path_tmp)
    return entitys, concept_paths, bit_index



if __name__ == "__main__":
    agents = TwoGLMAgents(key="your glm4 key")
    import jsonlines
    import numpy as np
    import jsonlines
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_retry", default=5, type=int)
    parser.add_argument("--semantic_tree_filepath", default="0328_refined_semantic_tree.json")
    parser.add_argument("--sentence_num", default=1000, type=int)
    args = parser.parse_args()

    from coding import Coding
    P_Coder = Coding(args.semantic_tree_filepath, ["人物", ], only_use_deepest=True)
    L_Coder = Coding(args.semantic_tree_filepath, ["位置", "房产"], only_use_deepest=False)
    # T_Coder = Coding(args.semantic_tree_filepath, ["时间"], only_use_deepest=False)
    np.random.seed(42)

    bits_stream = np.random.randint(high=2, low=0, size=(1, 100000)).tolist()[0]
    bits_stream += bits_stream
    bits_stream += bits_stream
    bits_stream += bits_stream
    bits_stream += bits_stream
    bit_index_old = np.random.randint(0, 10000)
    # keywords_map = {"P":"person", "L":"location", "T": "time"}
    # input_str_map = {"P":"人物", "L":"地点", "T":"时间"}
    import os
    os.makedirs("alg", exist_ok=True)

    for max_retry in range(3, args.max_retry+1):
        bit_index = bit_index_old
        idx = 0
        agents = TwoGLMAgents(key="26fe09d662776a62c47e279cc37235df.ScFgcBiT4pCNMuOW",)
        with jsonlines.open(f"alg/semantic_steg_output0402_AIAgent_maxretry{max_retry}_hc.jsonl", "w") as f_out:
            for _ in tqdm(range(args.sentence_num)):
                entitys, concept_paths, bit_num_used = encode([P_Coder,L_Coder], ["P", "L"],
                                                              bits_stream[bit_index:], return_single=False,
                                                              alg="hc")
                bits = bits_stream[bit_index:bit_index + bit_num_used]
                bit_index += bit_num_used

                keywords = {"person": entitys[0], "location": entitys[1]}
                input_str = f"人物集合{entitys[0][:10]}中的一个人物和地点集合{entitys[1][:10]}中的一个地点"
                output_str = f"人物：{entitys[0][0]}，地点：{entitys[1][0]}"

                try:
                    is_success, stego, extract_results, retry = agents.action(input_str, check_keys=keywords, max_retry=max_retry)
                    if is_success:
                        f_out.write({"idx":idx, "stego": stego, "retry": retry, "score": 1, "is_success":is_success, "bits": bits, "entitys":keywords})
                    else:
                        f_out.write({"idx":idx, "stego": output_str, "retry": retry, "score": 0, "is_success":is_success, "bits": bits, "entitys": keywords})
                except:
                    idx += 1

    for max_retry in range(3, args.max_retry+1):
        bit_index = bit_index_old
        idx = 0
        agents = TwoGLMAgents(key="26fe09d662776a62c47e279cc37235df.ScFgcBiT4pCNMuOW",)
        with jsonlines.open(f"alg/semantic_steg_output0402_AIAgent_check_maxretry{max_retry}_hc.jsonl", "w") as f_out:
            for _ in tqdm(range(args.sentence_num)):
                entitys, concept_paths, bit_num_used = encode([P_Coder,L_Coder], ["P", "L"],
                                                              bits_stream[bit_index:], return_single=False,
                                                              alg="hc")
                bits = bits_stream[bit_index:bit_index + bit_num_used]
                bit_index += bit_num_used

                keywords = {"person": entitys[0], "location": entitys[1]}
                input_str = f"人物集合{entitys[0][:10]}的一个人物和地点集合{entitys[1][:10]}的一个地点"
                output_str = f"人物：{entitys[0][0]}，地点：{entitys[1][0]}"

                try:
                    is_success, stego, extract_results, retry = agents.action_0223(input_str, check_keys=keywords, max_retry=max_retry)
                    if is_success:
                        f_out.write({"idx":idx, "stego": stego, "retry": retry, "score": 1, "is_success":is_success, "bits": bits, "entitys":keywords})
                    else:
                        f_out.write({"idx":idx, "stego": output_str, "retry": retry, "score": 0, "is_success":is_success, "bits": bits, "entitys": keywords})
                except:
                    idx += 1

    for max_retry in range(3, args.max_retry + 1):
        bit_index = bit_index_old
        idx = 0
        agents = TwoGLMAgents(key="26fe09d662776a62c47e279cc37235df.ScFgcBiT4pCNMuOW", )
        with jsonlines.open(f"alg/semantic_steg_output0402_AIAgent_maxretry{max_retry}_ac.jsonl", "w") as f_out:
            for _ in tqdm(range(args.sentence_num)):
                entitys, concept_paths, bit_num_used = encode([P_Coder, L_Coder], ["P", "L"],
                                                              bits_stream[bit_index:], return_single=False,
                                                              alg="ac")
                bits = bits_stream[bit_index:bit_index + bit_num_used]
                if bit_num_used == 0:
                    bit_num_used = 1
                bit_index += bit_num_used

                keywords = {"person": entitys[0], "location": entitys[1]}
                input_str = f"人物集合{entitys[0][:10]}中的一个人物和地点集合{entitys[1][:10]}中的一个地点"
                output_str = f"人物：{entitys[0][0]}，地点：{entitys[1][0]}"

                try:
                    is_success, stego, extract_results, retry = agents.action(input_str, check_keys=keywords,
                                                                              max_retry=max_retry)
                    if is_success:
                        f_out.write({"idx": idx, "stego": stego, "retry": retry, "score": 1, "is_success": is_success,
                                     "bits": bits, "entitys": keywords})
                    else:
                        f_out.write(
                            {"idx": idx, "stego": output_str, "retry": retry, "score": 0, "is_success": is_success,
                             "bits": bits, "entitys": keywords})
                except:
                    idx += 1

    for max_retry in range(3, args.max_retry + 1):
        bit_index = bit_index_old
        idx = 0
        agents = TwoGLMAgents(key="26fe09d662776a62c47e279cc37235df.ScFgcBiT4pCNMuOW", )
        with jsonlines.open(f"alg/semantic_steg_output0402_AIAgent_check_maxretry{max_retry}_ac.jsonl", "w") as f_out:
            for _ in tqdm(range(args.sentence_num)):
                entitys, concept_paths, bit_num_used = encode([P_Coder, L_Coder], ["P", "L"], bits_stream[bit_index:],
                                                              return_single=False, alg="ac")
                bits = bits_stream[bit_index:bit_index + bit_num_used]
                if bit_num_used == 0:
                    bit_num_used = 1
                bit_index += bit_num_used

                keywords = {"person": entitys[0], "location": entitys[1]}
                input_str = f"人物集合{entitys[0][:10]}的一个人物和地点集合{entitys[1][:10]}的一个地点"
                output_str = f"人物：{entitys[0][0]}，地点：{entitys[1][0]}"

                try:
                    is_success, stego, extract_results, retry = agents.action_0223(input_str, check_keys=keywords,
                                                                                   max_retry=max_retry)
                    if is_success:
                        f_out.write({"idx": idx, "stego": stego, "retry": retry, "score": 1, "is_success": is_success,
                                     "bits": bits, "entitys": keywords})
                    else:
                        f_out.write(
                            {"idx": idx, "stego": output_str, "retry": retry, "score": 0, "is_success": is_success,
                             "bits": bits, "entitys": keywords})
                except:
                    idx += 1

