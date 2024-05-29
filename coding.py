import json
import os
from math import *
import re
import pygtrie
import heapq
import numpy as np
from decimal import *
def msb_bits2int(bits):
    res = 0
    for i, bit in enumerate(bits[::-1]):
        res += bit * (2 ** i)
    return res


def msb_int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in strlist]


class Coding(object):
    def __init__(self, knowledgebase_filepath, use_top_concepts=["人物"], only_use_leaf=True, only_use_deepest=True):
        data = json.load(open(knowledgebase_filepath, "r"))
        self.tree = data["tree"]
        self.max_depth = len(list(data["tree"].keys()))
        self.entity2concept = data["entity2concept"]
        self.trie = pygtrie.CharTrie()
        self.only_use_leaf = only_use_leaf
        self.only_use_deepest = only_use_deepest
        for depth in range(self.max_depth, 0, -1):
            if depth == self.max_depth:
                for concept in self.tree[f"level{depth}"]:
                    self.tree[f"level{depth}"][concept]["entity_num"] = len(self.tree[f"level{depth}"][concept]["entitys"])
                    self.tree[f"level{depth}"][concept]["entity_ratio"] = 1.0
                    self.tree[f"level{depth}"][concept]["entity_num_of_sons"] = 0
            else:
                for concept in self.tree[f"level{depth}"]:
                    self.tree[f"level{depth}"][concept]["entity_num"] = len(self.tree[f"level{depth}"][concept]["entitys"])
                    entity_num_sons = 0
                    for son in self.tree[f"level{depth}"][concept]["sons"]:
                        entity_num_sons += (self.tree[f"level{depth+1}"][son]["entity_num"] + self.tree[f"level{depth+1}"][son]["entity_num_of_sons"])
                    self.tree[f"level{depth}"][concept]["entity_num_of_sons"] = entity_num_sons
                    self.tree[f"level{depth}"][concept]["entity_ratio"] = self.tree[f"level{depth}"][concept]["entity_num"] /(self.tree[f"level{depth}"][concept]["entity_num"] + entity_num_sons)

        # 对不参与编码的顶层概率及子概念和实体剪枝
        if self.only_use_leaf:
            for depth in range(1, self.max_depth):
                for concept in self.tree[f"level{depth}"]:
                    if self.only_use_deepest:
                        self.tree[f"level{depth}"][concept]["entity_num"] = 0
                        self.tree[f"level{depth}"][concept]["entitys"] = []
                    else:
                        if len(self.tree[f"level{depth}"][concept]["sons"]) != 0:
                            self.tree[f"level{depth}"][concept]["entity_num"] = 0
                            self.tree[f"level{depth}"][concept]["entitys"] = []
            for depth in range(self.max_depth-1, 0, -1):
                for concept in self.tree[f"level{depth}"]:
                    sons = self.tree[f"level{depth}"][concept]["sons"]
                    entity_num_sons = 0
                    for son in sons:
                        if self.tree[f"level{depth+1}"].get(son, False):
                            entity_num_sons += self.tree[f"level{depth+1}"][son]["entity_num"] + self.tree[f"level{depth+1}"][son]["entity_num_of_sons"]
                        else:
                            entity_num_sons += 0
                    self.tree[f"level{depth}"][concept]["entity_num_of_sons"] = entity_num_sons
                if self.only_use_deepest:
                    self.tree[f"level{depth}"] = {concept:self.tree[f"level{depth}"][concept] for concept in self.tree[f"level{depth}"] if self.tree[f"level{depth}"][concept]["entity_num_of_sons"]!=0}

        if use_top_concepts is not None:
            self.use_top_concepts = use_top_concepts
            self.tree["level1"] = {use_top_concept:self.tree["level1"][use_top_concept] for use_top_concept in use_top_concepts}
        else:
            self.use_top_concepts = [concept for concept in self.tree["level1"]]
        for depth in range(2, self.max_depth+1):
            avaiable_concepts = []
            for father in self.tree[f"level{depth-1}"]:
                sons = [son for son in self.tree[f"level{depth-1}"][father]["sons"] if self.tree[f"level{depth}"].get(son, False)]
                avaiable_concepts += sons
                self.tree[f"level{depth - 1}"][father]["sons"] = sons
            self.tree[f"level{depth}"] = {concept:self.tree[f"level{depth}"][concept] for concept in avaiable_concepts}
        avaiable_entitys = []
        for depth in range(self.max_depth, 0, -1):
            for concept in self.tree[f"level{depth}"]:
                avaiable_entitys += self.tree[f"level{depth}"][concept]["entitys"]
        self.entity2concept = {entity:self.entity2concept[entity] for entity in avaiable_entitys}

        # 构建前缀匹配字典
        for entity in self.entity2concept:
            self.trie[entity] = 1


    def encode(self, bits_stream,):
        ### Fix-Length Coding
        bit_num_used = 0
        cur_depth = 1
        cur_level = self.tree["level1"]
        while True:
            candicates_1 = [concept for concept in cur_level]
            candicates_2 = [concept for concept in cur_level if cur_level[concept]["entity_num"] > 0]
            candicates = candicates_1 + candicates_2 if cur_depth != self.max_depth else candicates_1
            bit_max = floor(log2(len(candicates)))
            embed_int = msb_bits2int(bits_stream[bit_num_used:bit_num_used+bit_max])
            bit_num_used += bit_max
            next_concept = candicates[embed_int]
            if embed_int >= len(candicates_1) or cur_depth == self.max_depth or len(cur_level[next_concept]["sons"]) == 0: # meaning use the entitys of non-leaf concept
                used_entitys = cur_level[next_concept]["entitys"]
                break
            else:
                sons = cur_level[next_concept]["sons"]
                cur_depth += 1
                cur_level = {son:self.tree[f"level{cur_depth}"][son] for son in sons}
        concept = self.entity2concept[used_entitys[0]]
        concept_path = [concept]
        for depth in range(cur_depth, 1, -1):
            concept = self.tree[f"level{depth}"][concept]["father"]
            concept_path.append(concept)
        concept_path = concept_path[::-1]
        return used_entitys, concept_path, bit_num_used

    def encode_ac(self, bits_stream,):
        ### Huffman Coding
        bit_index = 0
        num_bits_encoded = 0
        cur_depth = 1
        cur_level = self.tree["level1"]
        precision = 100
        Low_interval = Decimal(0)
        High_interval = Decimal(2 ** precision)
        while True:
            candicates_1 = [concept for concept in cur_level]
            candicates_2 = [concept for concept in cur_level if cur_level[concept]["entity_num"] > 0]
            candicates = candicates_1 + candicates_2 if cur_depth != self.max_depth else candicates_1
            if cur_depth != self.max_depth:
                probs1 = [cur_level[concept]["entity_num_of_sons"] for concept in cur_level]
            else:
                probs1 = [cur_level[concept]["entity_num"] for concept in cur_level]
            probs2 = [cur_level[concept]["entity_num"] for concept in cur_level if cur_level[concept]["entity_num"] > 0]
            probs = probs1 + probs2 if cur_depth != self.max_depth else probs1
            probs = np.array(probs)
            probs = probs / np.sum(probs)
            prob_dict = {i: float(p) for i, p in enumerate(probs.tolist())}

            if len(candicates) == 1:
                embed_int = 0
            else:
                bits_decimal = Decimal(msb_bits2int(bits_stream[:precision]))
                probs = [Decimal(prob) for prob in np.array([0]+probs.tolist())]
                probs = np.cumsum(probs)
                probs[-1] = Decimal(1)
                probs = [prob*(High_interval-Low_interval)+Low_interval for prob in probs]
                for i, prob in enumerate(probs[:-1]):
                    if bits_decimal >= prob and bits_decimal<probs[i+1]:
                        Low_interval = prob
                        High_interval = probs[i+1]
                        embed_int = i
                        break

            next_concept = candicates[embed_int]
            if embed_int >= len(candicates_1) or cur_depth == self.max_depth or len(cur_level[next_concept]["sons"]) == 0: # meaning use the entitys of non-leaf concept
                used_entitys = cur_level[next_concept]["entitys"]
                break
            else:
                sons = cur_level[next_concept]["sons"]
                cur_depth += 1
                cur_level = {son:self.tree[f"level{cur_depth}"][son] for son in sons}

        Low_interval_bits = msb_int2bits(int(Low_interval), precision)
        High_interval_bits = msb_int2bits(int(High_interval), precision)
        while num_bits_encoded < precision:
            if Low_interval_bits[num_bits_encoded] == High_interval_bits[num_bits_encoded]:
                num_bits_encoded += 1
            else:
                break
        bit_index += num_bits_encoded
        concept = self.entity2concept[used_entitys[0]]
        concept_path = [concept]
        for depth in range(cur_depth, 1, -1):
            concept = self.tree[f"level{depth}"][concept]["father"]
            concept_path.append(concept)
        concept_path = concept_path[::-1]
        return used_entitys, concept_path, bit_index

    def encode_hc(self, bits_stream,):
        ### Arithmetic Coding
        bit_index = 0
        cur_depth = 1
        cur_level = self.tree["level1"]
        while True:
            candicates_1 = [concept for concept in cur_level]
            candicates_2 = [concept for concept in cur_level if cur_level[concept]["entity_num"] > 0]
            candicates = candicates_1 + candicates_2 if cur_depth != self.max_depth else candicates_1
            if cur_depth != self.max_depth:
                probs1 = [cur_level[concept]["entity_num_of_sons"] for concept in cur_level]
            else:
                probs1 = [cur_level[concept]["entity_num"] for concept in cur_level]
            probs2 = [cur_level[concept]["entity_num"] for concept in cur_level if cur_level[concept]["entity_num"] > 0]
            probs = probs1 + probs2 if cur_depth != self.max_depth else probs1
            probs = np.array(probs)
            probs = probs / np.sum(probs)
            prob_dict = {i: float(p) for i, p in enumerate(probs.tolist())}
            if len(candicates) == 1:
                embed_int = 0
                num_bits_encoded = 0
            else:
                hf = HuffmanCoding()
                hf.make_heap(prob_dict)
                hf.merge_nodes()
                hf.make_codes()
                for hf_code in hf.reverse_mapping.keys():
                    if hf_code == "".join([str(b) for b in bits_stream[bit_index:bit_index + len(hf_code)]]):
                        num_bits_encoded = len(hf_code)
                        embed_int = hf.reverse_mapping[hf_code]
                        break
            bit_index += num_bits_encoded
            next_concept = candicates[embed_int]
            if embed_int >= len(candicates_1) or cur_depth == self.max_depth or len(
                    cur_level[next_concept]["sons"]) == 0:  # meaning use the entitys of non-leaf concept
                used_entitys = cur_level[next_concept]["entitys"]
                break
            else:
                sons = cur_level[next_concept]["sons"]
                cur_depth += 1
                cur_level = {son: self.tree[f"level{cur_depth}"][son] for son in sons}
        concept = self.entity2concept[used_entitys[0]]
        concept_path = [concept]
        for depth in range(cur_depth, 1, -1):
            concept = self.tree[f"level{depth}"][concept]["father"]
            concept_path.append(concept)
        concept_path = concept_path[::-1]
        return used_entitys, concept_path, bit_index


    def decode(self, entity):
        cur_concept = self.entity2concept.get(entity, self.entity2concept.get(self.match(self.parser(entity)), None))
        if cur_concept is None:
            return [], []
        for depth_end in range(self.max_depth, 0, -1):
            if self.tree[f"level{depth_end}"].get(cur_concept, False):
                break
        bit_stream = []
        concept_path = [cur_concept]
        for depth in range(depth_end, 0, -1):
            father_concept = self.tree[f"level{depth}"][cur_concept]["father"]
            if depth > 1:
                father_level = self.tree[f"level{depth-1}"]
                father_sons = father_level[father_concept]["sons"]
                cur_level = {son:self.tree[f"level{depth}"][son] for son in father_sons}
            else:
                cur_level = self.tree[f"level{depth}"]
            candicates_1 = [concept for concept in cur_level]
            candicates_2 = [concept for concept in cur_level if cur_level[concept]["entity_num"] > 0]
            candicates = candicates_1 + candicates_2 if depth != self.max_depth else candicates_1
            bit_max = floor(log2(len(candicates)))
            candicate_index = []
            for index, candicate in enumerate(candicates[:2**bit_max]):
                if cur_concept == candicate:
                    candicate_index.append(index)
            if len(candicate_index) == 0: ### 即前2**bit_max分支中没有当前概念，解码失败, 强行解码输出
                for index, candicate in enumerate(candicates[2**bit_max:]):
                    if cur_concept == candicate:
                        candicate_index.append(index)
            if depth == depth_end:
                bit_stream.append(msb_int2bits(candicate_index[-1], bit_max))
            else:
                bit_stream.append(msb_int2bits(candicate_index[0], bit_max))
            cur_concept = father_concept
            if depth != 1:
                concept_path.append(cur_concept)

        concept_path = concept_path[::-1]

        bit_stream_final = []
        for bit_stream_tmp in bit_stream[::-1]:
            bit_stream_final += bit_stream_tmp
        return bit_stream_final, concept_path


    def parser(self, dirty_string):
        clean_string = re.sub("[\d,\W]", "", re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]|\\（.*?\\）|\\<.*?\\>|\\《.*?\\》", "", dirty_string))
        return clean_string


    def match(self, entity):
        # TODO
        #  可以使用实体融合算法处理
        #  也可以使用nltk简单处理lexicon
        # prefix 前缀匹配
        for chr_id in range(len(entity), 0, -1):
            prefix = entity[:chr_id]
            if self.trie.get(prefix, False):
                return prefix

        for chr_id in range(len(entity), 0, -1):
            prefix = entity[:chr_id]
            if self.trie.has_node(prefix):
                candicates = self.trie.items(prefix=prefix, shallow=True)
                min_length = 100000
                shortest_candicate = ""
                for candicate, _ in candicates:
                    if len(candicate) < min_length:
                        min_length = len(candicate)
                        shortest_candicate = candicate
                return shortest_candicate

        # suffix
        for chr_id in range(len(entity)):
            suffix = entity[chr_id:]
            if self.trie.get(suffix, False):
                return suffix
        for chr_id in range(len(entity)):
            suffix = entity[chr_id:]
            if self.trie.has_node(suffix):
                candicates = self.trie.items(prefix=suffix, shallow=True)
                min_length = 100000
                shortest_candicate = ""
                for candicate, _ in candicates:
                    if len(candicate) < min_length:
                        min_length = len(candicate)
                        shortest_candicate = candicate
                return shortest_candicate
        # original
        return entity

    def search(self, father_node, ):
        pass

    def get_sons(self, father_node, level):
        if self.tree["level" + str(level)].get(father_node, False):
            sons = self.tree["level" + str(level)][father_node]["sons"]
            return sons
        else:
            return []

    def get_level_nodes(self, level):
        nodes = []
        for key in self.tree["level" + str(level)]:
            nodes.append(key)
        return nodes


class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        # defining comparators less_than and equals
        def __lt__(self, other):
            return self.freq < other.freq

        def __eq__(self, other):
            if (other == None):
                return False
            if (not isinstance(other, HeapNode)):
                return False
            return self.freq == other.freq

    # functions for compression:

    def make_frequency_dict(self, text):
        frequency = {}
        for character in text:
            if not character in frequency:
                frequency[character] = 0
            frequency[character] += 1
        return frequency

    def make_heap(self, frequency):
        for key in frequency:
            node = self.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = self.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if (root == None):
            return

        if (root.char != None):
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self, text):
        encoded_text = ""
        for character in text:
            encoded_text += self.codes[character]
        return encoded_text

    def pad_encoded_text(self, encoded_text):
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text
        return encoded_text

    def get_byte_array(self, padded_encoded_text):
        if (len(padded_encoded_text) % 8 != 0):
            print("Encoded text not padded properly")
            exit(0)

        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i + 8]
            b.append(int(byte, 2))
        return b

    def compress(self):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + ".bin"

        with open(self.path, 'r+') as file, open(output_path, 'wb') as output:
            text = file.read()
            text = text.rstrip()

            frequency = self.make_frequency_dict(text)
            self.make_heap(frequency)
            self.merge_nodes()
            self.make_codes()

            encoded_text = self.get_encoded_text(text)
            padded_encoded_text = self.pad_encoded_text(encoded_text)

            b = self.get_byte_array(padded_encoded_text)
            output.write(bytes(b))

        print("Compressed")
        return output_path

    """ functions for decompression: """

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)

        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-1 * extra_padding]

        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = ""

        for bit in encoded_text:
            current_code += bit
            if (current_code in self.reverse_mapping):
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ""

        return decoded_text

    def decompress(self, input_path):
        filename, file_extension = os.path.splitext(self.path)
        output_path = filename + "_decompressed" + ".txt"

        with open(input_path, 'rb') as file, open(output_path, 'w') as output:
            bit_string = ""

            byte = file.read(1)
            while (len(byte) > 0):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

            encoded_text = self.remove_padding(bit_string)

            decompressed_text = self.decode_text(encoded_text)

            output.write(decompressed_text)

        print("Decompressed")
        return output_path

