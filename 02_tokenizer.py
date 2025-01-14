import os
import json
import random
import re
import unicodedata

import jaconv
from tqdm import tqdm
from rich.pretty import pprint as print
from sudachipy import tokenizer
from sudachipy import dictionary
from tokenizers import Tokenizer
from tokenizers import models
from tokenizers import decoders
from tokenizers import trainers
from tokenizers import processors
from tokenizers import normalizers
from tokenizers import pre_tokenizers

PATHS = (
    # "dataset/pt/modern_bert_multilingual_ds_pt_zh.txt",
    # "dataset/pt/modern_bert_multilingual_ds_pt_zh_r18_pixiv.txt",
    # "dataset/pt/modern_bert_multilingual_ds_pt_en.txt",
    "dataset/pt/modern_bert_multilingual_ds_pt_en_r18_visual_novels.txt",
    # "dataset/pt/modern_bert_multilingual_ds_pt_jp.txt",
    # "dataset/pt/modern_bert_multilingual_ds_pt_jp_r18.txt",
    "dataset/pt/modern_bert_multilingual_ds_pt_jp_r18_rpg.txt",
    # "dataset/pt/modern_bert_multilingual_ds_pt_ko.txt",
    # "dataset/pt/modern_bert_multilingual_ds_pt_ko_web.txt",
    # "dataset/pt/modern_bert_multilingual_ds_zh_cc100.txt",
    # "dataset/pt/modern_bert_multilingual_ds_zh_cc100_tw.txt",
    # "dataset/pt/modern_bert_multilingual_ds_en_c4.txt",
    # "dataset/pt/modern_bert_multilingual_ds_jp_cc100.txt",
    # "dataset/pt/modern_bert_multilingual_ds_ko_cc100.txt",
)
LIMIT = 80 * 10000
SUDACHI = dictionary.Dictionary(dict = "full").create()

# 设置随机种子
random.seed(42)

# 清理文本
def cleanup(line: str, path: str) -> str:
    # 将空格以外的空白符都替换为空格
    # \t：制表符
    # \n：换行符
    # \r：回车符
    # \v：垂直制表符
    # \f：换页符
    # \u3000：全角空格
    line = re.sub(r"[\t\n\r\v\f\u3000]+", " ", line)

    # 将多个空格替换为单个空格
    line = re.sub(r" +", " ", line)

    # 移除非文本字符
    # LS（行分隔符，Line Separator，Unicode 码点为 U+2028）
    # PS（段分隔符，Paragraph Separator，Unicode 码点为 U+2029）
    line = re.sub(r"[\x00-\x1F\x7F\u2028\u2029]", "", line)

    if "en" in path:
        line = unicodedata.normalize("NFKC", line)
    elif "jp" in path:
        # Convert Half-width (Hankaku) Katakana to Full-width (Zenkaku) Katakana
        # kana (bool) – Either converting Kana or not.
        # ascii (bool) – Either converting ascii or not.
        # digit (bool) – Either converting digit or not.
        line = jaconv.hankaku2zenkaku(line, kana = True, ascii = False, digit = False)

        # Convert Full-width (Zenkaku) Katakana to Half-width (Hankaku) Katakana
        # kana (bool) – Either converting Kana or not.
        # ascii (bool) – Either converting ascii or not.
        # digit (bool) – Either converting digit or not.
        line = jaconv.zenkaku2hankaku(line, kana = False, ascii = True, digit = True)

    return line.strip()

def pre_tokenize(line: str, path: str) -> str:
    line = cleanup(line, path)

    if "jp" in path:
        return " ".join([m.surface() for m in SUDACHI.tokenize(line, tokenizer.Tokenizer.SplitMode.C)])

    return line

def main() -> None:
    # 数据抽样

    result = []
    # result = PATHS
    os.makedirs("tokenizer", exist_ok = True)
    for path in PATHS:
        output = f"tokenizer/{os.path.basename(path)}"
        if not os.path.isfile(output):
            with open(path, "r", encoding = "utf-8") as reader:
                lines = [line.strip() for line in reader if line.strip() != ""]
                lines = [pre_tokenize(line, path) for line in tqdm(random.sample(lines, min(LIMIT, len(lines))), desc = path)]
            with open(output, "w", encoding = "utf-8") as writer:
                writer.write("\n".join(lines))
        result.append(output)

    # 分词器配置
    trainer = trainers.UnigramTrainer(
        unk_token = "[UNK]",
        vocab_size = 32000 * 3,
        n_sub_iterations = 2,
    )
    tokenizer = Tokenizer(
        models.Unigram(),
    )
    tokenizer.add_special_tokens([
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[PAD]",
        "[MASK]",
    ])
    tokenizer.normalizer = normalizers.Sequence([])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Punctuation(behavior = "isolated"),
            pre_tokenizers.Digits(individual_digits = True),
            # pre_tokenizers.Metaspace(),
            pre_tokenizers.Whitespace(),
            # pre_tokenizers.ByteLevel(
            #     use_regex = False,
            #     add_prefix_space = False,
            # ),
        ]
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single = "[CLS]:0 $A [SEP]:0",
        pair = "[CLS]:0 $A [SEP]:0 $B [SEP]:0",
        special_tokens = [
            ("[UNK]", 50280),
            ("[CLS]", 50281),
            ("[SEP]", 50282),
            ("[PAD]", 50283),
            ("[MASK]", 50284),
        ],
    )
    tokenizer.decoder = decoders.ByteLevel()

    # 开始训练
    tokenizer.train(
        files = result,
        trainer = trainer,
    )

    # 保存结果
    tokenizer.save(
        "tokenizer_output.json",
        pretty = True,
    )

    # 不知道为什么，添加的特殊 Token 的 ID 不会生效
    # 所以手动修一下
    with open("tokenizer_output.json", "r", encoding = "utf-8") as reader:
        tokenizer = json.load(reader)

    pairs = {
        "[UNK]": 50280,
        "[CLS]": 50281,
        "[SEP]": 50282,
        "[PAD]": 50283,
        "[MASK]": 50284,
    }

    tokenizer["model"]["unk_id"] = pairs["[UNK]"]
    tokenizer["decoder"]["add_prefix_space"] = False

    for i, item in enumerate(tokenizer.get("added_tokens")):
        if item.get("content") in pairs:
            tokenizer["added_tokens"][i]["id"] = pairs[item.get("content")]
        print(tokenizer["added_tokens"][i])

    for k, v in pairs.items():
        tokenizer.get("model").get("vocab")[v - 50280] = tokenizer.get("model").get("vocab")[v]
        tokenizer.get("model").get("vocab")[v] = (k, 0.0)
        print(f"{tokenizer.get("model").get("vocab")[v - 50280]} -> {tokenizer.get("model").get("vocab")[v]}")

    with open("tokenizer/tokenizer.json", "w", encoding = "utf-8") as writer:
        json.dump(tokenizer, writer, ensure_ascii = False, indent = 4)

if __name__ == "__main__":
    main()