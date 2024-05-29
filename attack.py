#重写
def paraphrase(model, tokenizer,text, language="en", top_k = 15, temperature = 1.0, max_length = 50):
    if language == "en":
        prompt_pool ={
            "0":"paraphrase the following paragraphs:\n",
            "1":"paraphrase the following paragraphs and try your best not to use the same bigrams from the original paragraphs\n",
            "2":"paraphrase the following paragraphs and try to keep the similar length to the original paragraphs\n",
            "3":"You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. This is the text: \n",
            "4":"As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases."
        }
    elif language == "zh":
        prompt_pool ={
            "0":"改写以下段落:\n",
            "1":"改写以下段落，尽量不要使用与原段落相同的词语:\n",
            "2":"改写以下段落，并尽量保持与原始段落相似的长度:\n",
            "3":"你是一位专业的文案编辑。请用自己的风格改写下面的文字，并对所有句子进行转述。\n确保最终输出包含与原始文本相同的信息，并且长度大致相同。\n用自己的风格重写时，不要遗漏任何重要的细节。这是文本：\n",
            "4":"作为一名专业的文案编辑，请用自己的凤凤重写以下文本，同时确保最终输出包含与原文相同的信息，并且长度大致相同。请复述所有句子，不要遗漏任何关键细节。此外，请注意提供文本中提到的公众人物、组织或其他实体的任何相关信息，以避免任何潜在的误解或偏见。这是文本：\n",
        }
    else:
        prompt_pool ={
            "0":"paraphrase the following paragraphs:\n",
            "1":"paraphrase the following paragraphs and try your best not to use the same bigrams from the original paragraphs\n",
            "2":"paraphrase the following paragraphs and try to keep the similar length to the original paragraphs\n",
            "3":"You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. This is the text: \n",
            "4":"As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases."
        }
    prompt = prompt_pool["0"] + text
    print(prompt)
    response = model.generate(tokenizer.encode(prompt,return_tensors="pt").to("cuda"),
                        top_k=top_k,
                        temperature = temperature,
                        do_sample=True,
                        max_length=max_length
                        )
    return tokenizer.decode(response[0][len(tokenizer.encode(prompt,return_tensors="pt")[0]):])


#打乱顺序
def scramble(text):
    import random
    fragments = text.split(".")
    new_fragments = []
    for i in range(len(fragments)):
        if fragments[i] == '':
            continue
        else:
            new_fragments.append(fragments[i].strip())
    # print(fragments)
    random.shuffle(fragments)
    return ". ".join(fragments) + "."


#随机替换
def random_replace(tokenizer, text, attack_len = 1,):
    import torch
    input_ids = tokenizer.encode(text,return_tensors="pt")[0]
    # locations = [0,0]
    # while locations[0] == locations[1]:
    #     locations = torch.randint(low = 1, high = len(input_ids) - attack_len, size = (2,))
    locations = torch.randint(low=1, high=len(input_ids) - attack_len, size=(2,))
    input_ids[locations[0] : locations[0] + attack_len] = input_ids[locations[1]: locations[1] + attack_len]
    return tokenizer.decode(input_ids)


#随机插入
def random_insert(tokenizer, text,attack_len = 1,):
    import torch
    input_ids = tokenizer.encode(text,return_tensors="pt")[0]
    # locations = [0,0]
    # while locations[0] == locations[1]:
    locations = torch.randint(low = 1, high = len(input_ids) - attack_len, size = (2,))
    new_input_ids = input_ids[:locations[0]].tolist() + input_ids[locations[1]:locations[1]+attack_len].tolist() + input_ids[locations[0]:].tolist()
    return tokenizer.decode(new_input_ids)


#随机删除
def random_delete(tokenizer, text, attack_len = 1,):
    import torch
    input_ids = tokenizer.encode(text,return_tensors="pt")[0]
    # locations = [0,0]
    # while locations[0] == locations[1]:
    locations = torch.randint(low = 1, high = len(input_ids) - attack_len, size = (attack_len,))
    new_input_ids = input_ids.tolist()
    for i in locations:
        del new_input_ids[i]
    return tokenizer.decode(new_input_ids)


if __name__ == "main":
    # paraphrase(model,tokenizer,"i have a lamp on my desk")
    # scramble("hello. bro. hi.")
    # tokenizer.encode("hello. bro. hi.",return_tensors="pt")[0]
    # torch.randint(low = 0, high = 5, size = (2,))
    # random_insert(tokenizer,"hello there hello bri")
    # random_delete(tokenizer,"hello there hello bri")
    pass