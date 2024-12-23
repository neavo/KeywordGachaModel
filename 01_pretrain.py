import json
import os
import re
import random
import unicodedata
from datetime import datetime

import wandb
import torch
import jaconv
from tqdm import tqdm
from rich import print
from joblib import delayed
from joblib import Parallel
from datasets import Dataset
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import DataCollatorForWholeWordMask
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from model.PreTrainerCallback import PreTrainerCallback

# 模型
MODEL_NAME = "modern_bert_multilingual"
MODEL_PATH = f"assets/{MODEL_NAME}"
OUTPUT_PATH = f"output/{MODEL_NAME}_pt"

# 训练
LEARNING_RATE = 2 * 1e-5
EPOCHS = 2
EVAL_SIZE = 16
BATCH_SIZE = 16
GRADIENT_CHECKPOINTING = False
GRADIENT_ACCUMULATION_SIZE = 128

# 输出
LOG_STEPS = 5
INTERVAL_STEPS = 100
SAVE_TOTAL_LIMIT = 0
AUTO_RESUME_FROM_CHECKPOINT = True

# 数据
DO_LOWER_CASE = False
LENGTH_THRESHOLD = 256
DATASET_PATH = [
    ("dataset/pretrain/en", 20 * 10000),
    ("dataset/pretrain/en_r18_visual_novels", 20 * 10000),
    ("dataset/pretrain/zh", 20 * 10000),
    ("dataset/pretrain/zh_r18_pixiv", 20 * 10000),
    ("dataset/pretrain/jp", 50 * 10000),
    ("dataset/pretrain/jp_r18", 20 * 10000),
    ("dataset/pretrain/jp_r18_rpg", 10 * 10000),
    ("dataset/pretrain/kr", 40 * 10000),
]

# 加载分词器
def load_tokenizer() -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(
        MODEL_PATH,
        do_lower_case = DO_LOWER_CASE,
        local_files_only = True,
    )

# 分割数组
def split(datas: list[str], size: int) -> list[list[str]]:
    return [datas[i:(i + size)] for i in range(0, len(datas), size)]

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

    return line

# 生成数据
def generate_datas(tokenizer: PreTrainedTokenizerFast, lines: list[str], path: str) -> list[str]:
    lines = [cleanup(line, path) for line in lines]

    datas = []
    tokens = tokenizer(
        lines,
        padding = False,
        truncation = True,
        max_length = LENGTH_THRESHOLD,
    )

    for line, input_ids in zip(lines, tokens.input_ids):
        datas.append({
            "line": line,
            "length": len(input_ids),
        })

    return datas

# 生成数据块
def generate_chunks(tokenizer: PreTrainedTokenizerFast, lines: list[str], path: str) -> list[list[str]]:
    chunks = []
    datas = generate_datas(tokenizer, lines, path)

    chunk = ""
    chunk_length = 0
    for data in datas:
        line = data.get("line")
        length = data.get("length")

        if chunk_length + length >= LENGTH_THRESHOLD - 3:
            chunk = re.sub(r" +", " ", chunk + " " + line)
            chunks.append(chunk.strip())

            chunk = ""
            chunk_length = 0
        else:
            chunk = chunk + " " + line
            chunk_length = chunk_length + 1 + length - 2 # 空格算不算 Token 都有可能，保险起见 +1，再减去首尾的两个特殊 Token

    if chunk.strip() != "":
        chunk = re.sub(r" +", " ", chunk)
        chunks.append(chunk.strip())

    return chunks

# 准备语料
def generate_text_file(path: str, file_path: str, tokenizer: PreTrainedTokenizerFast, threshold: int) -> list[str]:
    # 读取所有文件
    lines = []
    total = len([f for f in os.scandir(path) if f.name.endswith(".txt")])
    for file in tqdm(os.scandir(path), desc = path, total = total):
        if file.name.endswith(".txt"):
            with open(file.path, "r", encoding = "utf-8") as file:
                total = total + 1
                lines.extend([line.strip() for line in file if line.strip() != ""])

    # 切割成数据分段
    chunks = split(lines, 32 * 1024)

    # 并行处理数据分段进行
    data = []
    results = Parallel(n_jobs = os.cpu_count(), prefer = "processes", return_as = "generator_unordered")(
        delayed(generate_chunks)(tokenizer, v, path) for v in chunks
    )
    for result in tqdm(results, desc = path, total = len(chunks)):
        data.extend(result)

    # 按阈值随机取数据，然后写入文件
    data = random.sample(data, min(int(threshold), len(data)))
    with open(file_path, "w", encoding = "utf-8") as file:
        file.writelines("\n".join(data))

    return total

# 加载数据集
def load_dataset(tokenizer: PreTrainedTokenizerFast) -> tuple[Dataset, Dataset]:
    print("")
    print("正在加载数据集 ...")
    print("")

    # 遍历数据集路径
    total = 0
    paths = []
    for path, threshold in DATASET_PATH:
        dir_path, dir_name = os.path.split(path)

        # 如果数据文本文件不存在，则生成
        file_path = f"{dir_path}/{MODEL_NAME}_{dir_name}.txt"
        if os.path.isfile(file_path) == True:
            total = total + 1
        else:
            total = generate_text_file(path, file_path, tokenizer, threshold)

        # 记录路径
        paths.append(file_path)

    # 生成数据集
    os.makedirs("dataset/pretrain/cache", exist_ok = True)
    dataset_tokenized = Dataset.from_text(paths).map(
        lambda samples: map_function(tokenizer, samples),
        num_proc = 8,
        batched = True,
        batch_size = 1024,
        writer_batch_size = 8 * 1024,
        remove_columns = ["text"],
        cache_file_name = f"dataset/pretrain/cache/{MODEL_NAME}.cache",
        load_from_cache_file = True,
    )

    # 计算有效的 Token 数量
    total_length = sum(dataset_tokenized["input_length"])

    # 打印数据集信息
    print(
        "\n"
        + f"找到数据文件 {total} 个，数据条目 {dataset_tokenized.num_rows} 个，"
        + f"有效 Token {(total_length / 1000 / 1000):.2f} M，平均每个条目 {(total_length / dataset_tokenized.num_rows):.2f} Token ..."
        + "\n"
    )

    # 拆分数据集
    dataset_dict = dataset_tokenized.train_test_split(
        seed = 42,
        shuffle = True,
        test_size = 2048,
        keep_in_memory = True,
        load_from_cache_file = False,
        test_cache_file_name = None,
        train_cache_file_name = None,
    )

    return dataset_dict.get("test"), dataset_dict.get("train")

# 映射函数
def map_function(tokenizer: PreTrainedTokenizerFast, samples: dict) -> BatchEncoding:
    encodings = tokenizer(
        samples["text"],
        padding = "max_length",
        truncation = True,
        max_length = LENGTH_THRESHOLD,
        return_attention_mask = True,
        return_offsets_mapping = True if tokenizer.is_fast else False, # 只有快速 tokenizer 才有这个功能
        return_special_tokens_mask = True,
    )

    # 计算有效的 Token 数量
    encodings["input_length"] = [sum(item) for item in encodings.attention_mask]

    return encodings

# 加载模型
def load_model() -> PreTrainedModel:
    return AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH,
        local_files_only = True,
        trust_remote_code = True,
        ignore_mismatched_sizes = True,
        torch_dtype = torch.bfloat16,
        attn_implementation = "flash_attention_2",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

# 打印模型的参数量
def print_model_parameters(model: PreTrainedModel) -> None:
    total = 0
    layer = 0
    embedding = 0
    for name, param in model.named_parameters():
        total = total + param.numel()
        if "embeddings" not in name:
            layer = layer + param.numel()
        else:
            embedding = embedding + param.numel()

    print("")
    print(f"{MODEL_NAME} : layer - {layer / 1e6:.2f} M / embedding - {embedding / 1e6:.2f} M / total - {total / 1e6:.2f} M")
    print("")

# 开始训练
def start_training(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, eval_dataset: Dataset, train_dataset: Dataset) -> None:
    training_args = TrainingArguments(
        # 输出
        report_to = "wandb",
        output_dir = OUTPUT_PATH,
        logging_dir = "logs",
        logging_steps = LOG_STEPS,
        eval_steps = INTERVAL_STEPS,
        save_steps = INTERVAL_STEPS,
        eval_strategy = "steps",
        save_strategy = "steps",
        save_total_limit = SAVE_TOTAL_LIMIT,

        # 训练
        bf16 = True,
        bf16_full_eval = True,
        optim = "ademamix_8bit",
        warmup_ratio = 0.1,
        weight_decay = 0.01,
        learning_rate = LEARNING_RATE,
        num_train_epochs = EPOCHS,
        lr_scheduler_type = "cosine",
        per_device_eval_batch_size = EVAL_SIZE,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_checkpointing = GRADIENT_CHECKPOINTING,
        gradient_accumulation_steps = max(1, int(GRADIENT_ACCUMULATION_SIZE / BATCH_SIZE)),
    )

    trainer = Trainer(
        args = training_args,
        model = model,
        callbacks = [
            PreTrainerCallback(),
        ],
        data_collator = DataCollatorForWholeWordMask(
            tokenizer = tokenizer,
            mlm = True,
            mlm_probability = 0.30
        ),
        eval_dataset = eval_dataset,
        train_dataset = train_dataset,
        processing_class = tokenizer,
    )

    resume_from_checkpoint = (
        AUTO_RESUME_FROM_CHECKPOINT == True
        and any(v.startswith("checkpoint") and os.path.isdir(f"{OUTPUT_PATH}/{v}") for v in os.listdir(OUTPUT_PATH))
    )

    if resume_from_checkpoint:
        print(f"在 {OUTPUT_PATH} 找到可恢复的训练状态，自动继续训练 ...")

    trainer.train(
        resume_from_checkpoint = resume_from_checkpoint,
    )

# 主函数
def main() -> None:
    # 加载分词器
    tokenizer = load_tokenizer()

    # 加载数据集
    eval_dataset, train_dataset = load_dataset(tokenizer)

    # 加载模型
    model = load_model()

    # 调整 token_embeddings 的大小
    model.resize_token_embeddings(len(tokenizer))

    # 打印模型的参数量
    print_model_parameters(model)

    # 设置 wandb
    wandb.init(
        project = "PRETRAIN",
        name = f"{MODEL_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}",
    )

    # 开始训练
    start_training(model, tokenizer, eval_dataset, train_dataset)

# 主函数
if __name__ == "__main__":
    main()