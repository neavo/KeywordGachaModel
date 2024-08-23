import os
import re
import json
import random
from datetime import datetime

import wandb
import torch
import bitsandbytes
from tqdm import tqdm
from rich import print
from joblib import delayed
from joblib import Parallel
from datasets import Dataset
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForWholeWordMask
from transformers import DataCollatorForLanguageModeling

from helper.TextHelper import TextHelper
from model.PreTrainerCallback import PreTrainerCallback

# 参数设置
MODEL_NAME = "microsoft_mdeberta_v3_base"
MODEL_PATH = f"assets/{MODEL_NAME}"
OUTPUT_PATH = f"output/{MODEL_NAME}_pretrain"
EPOCHS = 2
MAX_LENGTH = 512
BATCH_SIZE = 6
GRADIENT_ACCUMULATION_SIZE = 64
DO_LOWER_CASE = False
LEARNING_RATE = 2 * 1e-5
INTERVAL_STEPS = 200
AUTO_RESUME_FROM_CHECKPOINT = True

DATASET_PATH = [
    ("dataset/pretrain/en", 10 * 10000),
    ("dataset/pretrain/en_r18_visual_novels", 10 * 10000),
    ("dataset/pretrain/zh", 10 * 10000),
    ("dataset/pretrain/zh_r18_pixiv", 10 * 10000),
    ("dataset/pretrain/jp", 8.5 * 10000),
    ("dataset/pretrain/jp_r18", 8.5 * 10000),
    ("dataset/pretrain/jp_r18_rpgmaker", 3 * 10000),
]

# 加载分词器
def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_PATH,
        do_lower_case = DO_LOWER_CASE,
        local_files_only = True,
    )

# 清理文本
def cleanup(line):
    # 【\N[123]】 这种形式是代指角色名字的变量
    # 直接抹掉就没办法判断角色了，只把 \N 部分抹掉，保留 ID 部分
    line = line.strip().replace("\\N", "")

    # 放大或者缩小字体的代码，干掉
    # \{\{ゴゴゴゴゴゴゴゴゴッ・・・\r\n（大地の揺れる音）
    line = re.sub(r"(\\\{)|(\\\})", "", line) 

    # /C[4] 这种形式的代码，干掉
    line = re.sub(r"/[A-Z]{1,5}\[\d+\]", "", line, flags = re.IGNORECASE)

    # \FS[29] 这种形式的代码，干掉
    line = re.sub(r"\\[A-Z]{1,5}\[\d+\]", "", line, flags = re.IGNORECASE)

    # \nw[隊員Ｃ] 这种形式的代码，干掉 [ 前的部分
    line = re.sub(r"\\[A-Z]{1,5}\[", "[", line, flags = re.IGNORECASE)

    # 由于上面的代码移除，可能会产生空人名框的情况，干掉
    line = line.replace("【】", "") 

    # 干掉除了空格以外的行内空白符（包括换行符、制表符、回车符、换页符等）
    line = re.sub(r"[^\S ]+", "", line)

    # 合并连续的空格为一个空格
    line = re.sub(r" +", " ", line)

    # 移除开头结尾的符号
    line = TextHelper.strip_punctuation(line)

    return line

# 获取长度
def get_length(tokenizer, line):
    return len(
        tokenizer(line).input_ids
    )

# 生成数据
def generate_datas(tokenizer, lines):
    lines = [cleanup(line) for line in lines]

    datas = []
    tokens = tokenizer(
        lines,
        padding = False,
        truncation = True, 
        max_length = MAX_LENGTH,
    )

    for line, input_id in zip(lines, tokens.input_ids):
        datas.append({
            "line": line,
            "length": len(input_id),
        })

    return datas

# 生成数据块
def generate_chunks(tokenizer, lines):
    chunks = []
    datas = generate_datas(tokenizer, lines)

    longest = 0
    chunk = ""
    chunk_length = 0
    for data in datas:
        line = data.get("line")
        length = data.get("length")

        # 如果有乱码，跳过
        if "�" in line:
            continue

        # 单句长度就超过最大值，则跳过
        if length > MAX_LENGTH - 3:
            continue

        if chunk_length + length > MAX_LENGTH - 3:
            longest = max(longest, get_length(tokenizer, re.sub(r" +", " ", chunk.strip())))
            chunks.append(re.sub(r" +", " ", chunk.strip()))
            chunk = ""
            chunk_length = 0
            
        chunk = chunk + " " + line
        chunk_length = chunk_length + length + 1
    
    if chunk.strip() != "":
        longest = max(longest, get_length(tokenizer, re.sub(r" +", " ", chunk.strip())))
        chunks.append(re.sub(r" +", " ", chunk.strip()))

    return chunks, longest

# 加载数据集
def load_dataset(tokenizer):
    print(f"")
    print(f"正在加载数据集 ...")

    print(f"")
    longest = 0
    count = 0
    datas = []
    for path, num in DATASET_PATH:
        datas_by_type = []

        if os.path.exists(f"{path}_{MODEL_NAME}.txt"):
            count = count + 1
            with open(f"{path}_{MODEL_NAME}.txt", "r", encoding = "utf-8") as file:
                datas_by_type = [line.strip() for line in file]
        else:
            total = len([entry for entry in os.listdir(path) if os.path.isfile(os.path.join(path, entry))])

            lines = []
            for file in tqdm(os.scandir(path), desc = path, total = total):
                if file.name.endswith(".txt"):
                    with open(file.path, "r", encoding = "utf-8") as file:
                        count = count + 1
                        lines.extend([line.strip() for line in file if line.strip() != ""])

            lines = [lines[i:(i + 64 * 1024)] for i in range(0, len(lines), 64 * 1024)]
            results = Parallel(n_jobs = -1, prefer = "processes", return_as = "generator_unordered")(
                delayed(generate_chunks)(tokenizer, v) for v in lines
            )

            for v in tqdm(results, desc = path, total = len(lines)):
                datas_by_type.extend(v[0])
                longest = max(longest, v[1])

            datas_by_type = random.sample(datas_by_type, min(int(num), len(datas_by_type)))
            with open(f"{path}_{MODEL_NAME}.txt", "w", encoding = "utf-8") as file:
                file.writelines([f"{line}\n" for line in datas_by_type])

        datas.extend(datas_by_type)
    print(f"")
    print(f"最大长度为 {longest} ...") if longest > 0 else None
    print(f"找到数据文件 {count} 个，共 {len(datas)} 条数据 ...")

    # 生成数据集
    print(f"")
    os.makedirs("cache", exist_ok = True)
    dataset_train = Dataset.from_dict({"line": datas})
    dataset_train_tokenized = dataset_train.map(
        lambda samples: tokenizer(
            samples["line"], 
            padding = "max_length", 
            truncation = True, 
            max_length = MAX_LENGTH,
            return_attention_mask = True,
            return_offsets_mapping = True,
            return_special_tokens_mask = True,
        ),
        batched = True,
        batch_size = 1024,
        writer_batch_size = 4096,
        remove_columns = ["line"],
        cache_file_name = f"cache/{MODEL_NAME}.cache",
        load_from_cache_file = True,
    )
    print(f"")
    
    return dataset_train_tokenized

# 加载模型
def load_model():
    return AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH,
        local_files_only = True,
        ignore_mismatched_sizes = True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

# 打印模型的参数量
def print_model_parameters(model):
    total = 0
    layer = 0
    embedding = 0
    for name, param in model.named_parameters():
        total = total + param.numel()
        if not "embeddings" in name:
            layer = layer + param.numel()
        else:
            embedding = embedding + param.numel()

    print(f"")
    print(f"{MODEL_NAME} : layer - {layer / 1e6:.2f} M / embedding - {embedding / 1e6:.2f} M / total - {total / 1e6:.2f} M")
    print(f"")

# 设置 wandb
def set_wandb():
    wandb.require("core")
    wandb.init(
        project = f"PRETRAIN",
        name = f"{MODEL_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}",
    )

# 开始训练
def start_training(model, tokenizer, dataset_train_tokenized):
    training_args = TrainingArguments(
        optim = "adamw_8bit",
        output_dir = OUTPUT_PATH,
        warmup_ratio = 0.1,
        weight_decay = 0.01,
        learning_rate = LEARNING_RATE,
        logging_dir = "logs",   
        logging_steps = INTERVAL_STEPS / 10,     
        eval_strategy = "no",
        save_strategy = "steps",
        save_steps = INTERVAL_STEPS,
        save_total_limit = 3,
        save_safetensors = False,
        num_train_epochs = EPOCHS,
        bf16 = True,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_checkpointing = True,
        gradient_accumulation_steps = max(1, int(GRADIENT_ACCUMULATION_SIZE / BATCH_SIZE)),
    )

    trainer = Trainer(
        args = training_args,
        model = model,
        tokenizer = tokenizer,
        callbacks = [
            PreTrainerCallback(),
        ],
        optimizers = (
            bitsandbytes.optim.Adam8bit(
                model.parameters(),
                lr = LEARNING_RATE,
                weight_decay = 0.01,
            ),
            None,
        ),
        data_collator = DataCollatorForWholeWordMask(
            tokenizer = tokenizer,
            mlm = True, 
            mlm_probability = 0.15
        ),
        train_dataset = dataset_train_tokenized,
    )

    resume_from_checkpoint = (
        AUTO_RESUME_FROM_CHECKPOINT and
        any(v.startswith("checkpoint") and os.path.isdir(f"{OUTPUT_PATH}/{v}") for v in os.listdir(OUTPUT_PATH))
    )

    if resume_from_checkpoint:
        print(f"在 {OUTPUT_PATH} 找到可恢复的训练状态，自动继续训练 ...")

    trainer.train(
        resume_from_checkpoint = resume_from_checkpoint,
    )

# 主函数
def main():
    # 加载分词器
    tokenizer = load_tokenizer()

    # 加载数据集
    dataset_train_tokenized = load_dataset(tokenizer)

    # 加载模型
    model = load_model()

    # 打印模型的参数量
    print_model_parameters(model)

    # 设置 wandb
    set_wandb()

    # 开始训练
    start_training(model, tokenizer, dataset_train_tokenized)

# 主函数
if __name__ == "__main__":
    main()