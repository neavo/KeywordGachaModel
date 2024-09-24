import os
import re
import sys

from tqdm import tqdm
from rich import print
from joblib import delayed
from joblib import Parallel

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class NERDataset(Dataset):

    CHUNK_SIZE = 2048

    def __init__(self, datas, tokenizer):
        self.tokenizer = tokenizer
        self.id2label, self.label2id = self.generate_id_label_map(datas)
        self.encodings, self.max_lenght, self.token_length_threshold = self.generate_encodings(datas)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]

    # 通过字符位置反查 token 位置 
    def char_to_token(self, encoding, char_start, char_end):
        token_end = 0
        token_start = 0

        for i, (start, end) in enumerate(encoding.offset_mapping):
            if start <= char_end < end:
                token_end = i
                break
    
        for i, (start, end) in enumerate(encoding.offset_mapping):
            if start <= char_start < end:
                token_start = i
                break

        return token_start, token_end

    # 获取 Token 长度阈值
    def get_token_length_threshold(self, datas):
        return max(len(self.tokenizer(data.get("sentence", "")).input_ids) for data in datas)

    # 生成数据块
    def generate_chunks(self, datas, token_length_threshold):
        encodings = []

        for data in datas:
            sentence = data.get("sentence", "")
            entities = data.get("entities", [])

            # 获取当前批次实体的名字的列表
            names = [entity.get("name", "") for entity in entities]

            # 获取当前批次实体的类型的字典
            ner_types = {entity.get("name", "") : entity.get("ner_type", "") for entity in entities}

            # 执行编码
            encoding = self.tokenizer(
                sentence, 
                padding = "max_length",
                truncation = True,
                max_length = token_length_threshold,
                return_offsets_mapping = True if self.tokenizer.is_fast else False, # 只有快速 tokenizer 才有这个功能
                return_special_tokens_mask = True
            )

            # 根据特殊标记设置 attention_mask
            for token_i, is_special_token in enumerate(encoding.special_tokens_mask):
                encoding.attention_mask[token_i] = 0 if is_special_token == 1 else 1

            # 根据 name 与 ner_type 查找位置并设置 labels
            if not self.tokenizer.is_fast:
                encoding.labels = self.generate_labels(
                    encoding, sentence, names, self.tokenizer.convert_ids_to_tokens(encoding.input_ids), ner_types
                )
            else:
                encoding.labels = self.generate_labels_fast(
                    encoding, sentence, names, encoding.tokens(), ner_types
                )            

            # 调试用
            # print(f"{sentence}")
            # print(f"{names}")
            # print(f"{encoding.labels}")
            # print(f"{self.tokenizer.convert_ids_to_tokens(encoding.input_ids)}")
            # raise

            # Trainer 会自动将数据移动到 GPU，不需要手动显式移动
            data = {}
            if hasattr(encoding, "labels") and encoding.labels is not None:
                data["labels"] = torch.tensor(encoding.labels)
            if hasattr(encoding, "input_ids") and encoding.input_ids is not None:
                data["input_ids"] = torch.tensor(encoding.input_ids)
            if hasattr(encoding, "token_type_ids") and encoding.token_type_ids is not None:
                data["token_type_ids"] = torch.tensor(encoding.token_type_ids)
            if hasattr(encoding, "attention_mask") and encoding.attention_mask is not None:
                data["attention_mask"] = torch.tensor(encoding.attention_mask)
            encodings.append(data)

        return encodings

    # 生成编码数据   
    def generate_encodings(self, datas):
        encodings = []
        max_lenght = 0

        # 分割数据
        datas = [datas[i:(i + self.CHUNK_SIZE)] for i in range(0, len(datas), self.CHUNK_SIZE)]

        # 获取 Token 长度阈值
        results = Parallel(n_jobs = -1, prefer = "processes", return_as = "generator_unordered")(
            delayed(self.get_token_length_threshold)(v) for v in datas
        )

        for v in tqdm(results, total = len(datas)):
            max_lenght = max(max_lenght, v)

        token_length_threshold = max_lenght + 4

        # 生成 Token 编码数据
        results = Parallel(n_jobs = -1, prefer = "processes", return_as = "generator_unordered")(
            delayed(self.generate_chunks)(v, token_length_threshold) for v in datas
        )

        for v in tqdm(results, total = len(datas)):
            encodings.extend(v)

        return encodings, max_lenght, token_length_threshold

    # 生成 ID-Label 映射表
    def generate_id_label_map(self, datas):
        ner_types = set()
        for data in datas:
            for entity in data.get("entities", []):
                ner_types.add(entity["ner_type"])

        id2label = {0: "O"}
        for c in list(sorted(ner_types)):
            id2label[len(id2label)] = f"B-{c}"
            id2label[len(id2label)] = f"I-{c}"
        label2id = {v: k for k, v in id2label.items()}

        return id2label, label2id

    # 生成 type_ids
    def generate_labels(self, encoding, sentence, names, tokens, ner_types):
        found_targets = []

        # 移除特殊 token 并记录实际 token 的起始位置
        special_tokens_mask = encoding.get("special_tokens_mask", [])
        actual_tokens = []
        token_position_map = []
        position = 0

        for i, token in enumerate(tokens):
            if special_tokens_mask[i] == 0:  # 非特殊 token
                actual_tokens.append(token)
                token_position_map.append(position)
                position += len(token.replace("##", ""))

        # 从每个 token 开始尝试匹配名字
        for name in names:
            for match in re.finditer(re.escape(name), sentence):
                char_start, char_end = match.start(), match.end()
                token_start, token_end = None, None

                for i, token in enumerate(actual_tokens):
                    token_start_offset = token_position_map[i]
                    token_end_offset = token_start_offset + len(token.replace("##", ""))

                    if token_start is None and token_start_offset <= char_start < token_end_offset:
                        token_start = i + 1  # 修正起始位置为实际 token 的位置
                        
                    if token_start is not None and token_start_offset < char_end <= token_end_offset:
                        token_end = i + 2  # 修正结束位置为实际 token 的位置
                        break

                if token_start is not None and token_end is not None:                    
                    found_targets.append({
                        "name": name,
                        "char_start": char_start,
                        "char_end": char_end,
                        "token_start": token_start,
                        "token_end": token_end,
                    })

        # 生成实际的 type_ids
        labels = [0 for _ in range(len(encoding.get("input_ids", [])))]
        for i in range(len(encoding.get("input_ids", []))):
            for target in found_targets:
                if target.get("token_start") == i:
                    labels[i] = self.label2id.get(f"B-{ner_types.get(target.get("name", ""), "O")}", 0)
                elif target.get("token_start") < i < target.get("token_end"):
                    labels[i] = self.label2id.get(f"I-{ner_types.get(target.get("name", ""), "O")}", 0)
        
        return labels

    # 生成标签列表 - 快速 Tokenizer 版本
    def generate_labels_fast(self, encoding, sentence, names, tokens, ner_types):
        found_targets = []

        # SentencePiece 有可能向句子开头添加内容为 _ 的 token
        # 这个 token 的值并不是特殊 token 应有的 (0, 0) 而是 (0, 1)
        # 这会干扰 char_to_token 的正常运作，所以需要先对 offset_mapping 进行矫正
        for k, v in enumerate(encoding.offset_mapping):
            if v[0] > 0:
                break
        for i in range(1, k - 1):
            encoding.offset_mapping[i] = (0, 0)

        for name in names:
            for match in re.finditer(re.escape(name), sentence):
                char_start, char_end = match.start(), match.end()
                token_start, token_end = self.char_to_token(encoding, char_start, char_end)

                found_targets.append({
                    "name": name,
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_start": token_start,
                    "token_end": token_end,
                })

        labels = [0 for _ in range(len(encoding.get("input_ids", [])))]
        for i in range(len(encoding.get("input_ids", []))):
            for target in found_targets:
                if target.get("token_start") == i:
                    labels[i] = self.label2id.get(f"B-{ner_types.get(target.get("name", ""), "O")}", 0)
                elif target.get("token_start") < i < target.get("token_end"):
                    labels[i] = self.label2id.get(f"I-{ner_types.get(target.get("name", ""), "O")}", 0)

        return labels