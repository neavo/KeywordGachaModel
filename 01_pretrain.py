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

# 参数设置
MODEL_NAME = "facebookai_xlm_roberta_base"
MODEL_PATH = f"assets/{MODEL_NAME}"
OUTPUT_PATH = f"output/{MODEL_NAME}_pretrain"
EPOCHS = 2
LENGTH_THRESHOLD = 256
BATCH_SIZE = 8
GRADIENT_CHECKPOINTING = False
GRADIENT_ACCUMULATION_SIZE = 0
DO_LOWER_CASE = False
LEARNING_RATE = 2 * 1e-5
INTERVAL_STEPS = 100
AUTO_RESUME_FROM_CHECKPOINT = True

DATASET_PATH = [
    ("dataset/pretrain/en", 60 * 10000),
    ("dataset/pretrain/en_r18_visual_novels", 20 * 10000),
    ("dataset/pretrain/zh", 40 * 10000),
    ("dataset/pretrain/zh_r18_pixiv", 40 * 10000),
    ("dataset/pretrain/jp", 100 * 10000),
    ("dataset/pretrain/jp_r18", 40 * 10000),
    ("dataset/pretrain/jp_r18_rpg", 20 * 10000),
    ("dataset/pretrain/kr", 80 * 10000),
]

# 可能存在的空字符
SPACE_PATTERN = r"\s*"

# 用于英文的代码段规则
CODE_PATTERN_EN = (
    SPACE_PATTERN + r"if\(.{0,5}[vs]\[\d+\].{0,10}\)" + SPACE_PATTERN,            # if(!s[982]) if(s[1623]) if(v[982] >= 1)
    SPACE_PATTERN + r"en\(.{0,5}[vs]\[\d+\].{0,10}\)" + SPACE_PATTERN,            # en(!s[982]) en(v[982] >= 1)
    SPACE_PATTERN + r"[/\\][a-z]{1,5}<[\d]{0,10}>" + SPACE_PATTERN,               # /C<1> \FS<12>
    SPACE_PATTERN + r"[/\\][a-z]{1,5}\[[\d]{0,10}\]" + SPACE_PATTERN,             # /C[1] \FS[12]
    SPACE_PATTERN + r"[/\\][a-z]{1,5}(?=<[^0-9]{0,10}>)" + SPACE_PATTERN,         # /C<非数字> \FS<非数字> 中的前半部分
    SPACE_PATTERN + r"[/\\][a-z]{1,5}(?=\[[^0-9]{0,10}\])" + SPACE_PATTERN,       # /C[非数字] \FS[非数字] 中的前半部分
)

# 用于非英文的代码段规则
CODE_PATTERN_NON_EN = (
    SPACE_PATTERN + r"if\(.{0,5}[vs]\[\d+\].{0,10}\)" + SPACE_PATTERN,            # if(!s[982]) if(v[982] >= 1) if(v[982] >= 1)
    SPACE_PATTERN + r"en\(.{0,5}[vs]\[\d+\].{0,10}\)" + SPACE_PATTERN,            # en(!s[982]) en(v[982] >= 1)
    SPACE_PATTERN + r"[/\\][a-z]{1,5}<[a-z\d]{0,10}>" + SPACE_PATTERN,            # /C<y> /C<1> \FS<xy> \FS<12>
    SPACE_PATTERN + r"[/\\][a-z]{1,5}\[[a-z\d]{0,10}\]" + SPACE_PATTERN,          # /C[x] /C[1] \FS[xy] \FS[12]
    SPACE_PATTERN + r"[/\\][a-z]{1,5}(?=<[^a-z0-9]{0,10}>)" + SPACE_PATTERN,      # /C<非数字非字母> \FS<非数字非字母> 中的前半部分
    SPACE_PATTERN + r"[/\\][a-z]{1,5}(?=\[[^a-z0-9]{0,10}\])" + SPACE_PATTERN,    # /C[非数字非字母] \FS[非数字非字母] 中的前半部分
)

# 同时作用于英文于非英文的代码段规则
CODE_PATTERN_COMMON = (
    SPACE_PATTERN + r"\\fr" + SPACE_PATTERN,                                      # 重置文本的改变
    SPACE_PATTERN + r"\\fb" + SPACE_PATTERN,                                      # 加粗
    SPACE_PATTERN + r"\\fi" + SPACE_PATTERN,                                      # 倾斜
    SPACE_PATTERN + r"\\\{" + SPACE_PATTERN,                                      # 放大字体 \{
    SPACE_PATTERN + r"\\\}" + SPACE_PATTERN,                                      # 缩小字体 \}
    SPACE_PATTERN + r"\\g" + SPACE_PATTERN,                                       # 显示货币 \G
    SPACE_PATTERN + r"\\\$" + SPACE_PATTERN,                                      # 打开金币框 \$
    SPACE_PATTERN + r"\\\." + SPACE_PATTERN,                                      # 等待0.25秒 \.
    SPACE_PATTERN + r"\\\|" + SPACE_PATTERN,                                      # 等待1秒 \|
    SPACE_PATTERN + r"\\!" + SPACE_PATTERN,                                       # 等待按钮按下 \!
    SPACE_PATTERN + r"\\>" + SPACE_PATTERN,                                       # 在同一行显示文字 \>
    # SPACE_PATTERN + r"\\<" + SPACE_PATTERN,                                     # 取消显示所有文字 \<
    SPACE_PATTERN + r"\\\^" + SPACE_PATTERN,                                      # 显示文本后不需要等待 \^
    # SPACE_PATTERN + r"\\n" + SPACE_PATTERN,                                     # 换行符 \\n
    SPACE_PATTERN + r"\r\n" + SPACE_PATTERN,                                      # 换行符 \r\n
    SPACE_PATTERN + r"\n" + SPACE_PATTERN,                                        # 换行符 \n
    SPACE_PATTERN + r"\\\\<br>" + SPACE_PATTERN,                                  # 换行符 \\<br>
    SPACE_PATTERN + r"<br>" + SPACE_PATTERN,                                      # 换行符 <br>
)

PATTERN_EN = re.compile(rf"(?:{"|".join(CODE_PATTERN_EN + CODE_PATTERN_COMMON)})+", re.IGNORECASE)
PATTERN_NON_EN = re.compile(rf"(?:{"|".join(CODE_PATTERN_NON_EN + CODE_PATTERN_COMMON)})+", re.IGNORECASE)

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
    if "en" in path:
        line = PATTERN_EN.sub(" ", line)
    else:
        line = PATTERN_NON_EN.sub(" ", line)

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
    results = Parallel(n_jobs = os.cpu_count() - 1, prefer = "processes", return_as = "generator_unordered")(
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
def load_dataset(tokenizer: PreTrainedTokenizerFast) -> Dataset:
    print(f"")
    print(f"正在加载数据集 ...")
    print(f"")

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
    dataset_train_tokenized = Dataset.from_text(paths).map(
        lambda samples: map_function(tokenizer, samples),
        num_proc = os.cpu_count() - 1,
        batched = True,
        batch_size = 1024,
        writer_batch_size = 8 * 1024,
        remove_columns = ["text"],
        cache_file_name = f"dataset/pretrain/cache/{MODEL_NAME}.cache",
        load_from_cache_file = True,
    )

    # 计算有效的 Token 数量
    total_length = sum(dataset_train_tokenized["input_length"])

    # 打印数据集信息
    print(
        "\n"
        + f"找到数据文件 {total} 个，数据条目 {dataset_train_tokenized.num_rows} 个，"
        + f"有效 Token {(total_length / 1000 / 1000):.2f} M，平均每个条目 {(total_length / dataset_train_tokenized.num_rows):.2f} Token ..."
        + "\n"
    )

    return dataset_train_tokenized

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
def start_training(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, dataset_train_tokenized: Dataset) -> None:
    training_args = TrainingArguments(
        optim = "ademamix_8bit",
        report_to = "wandb",
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
        num_train_epochs = EPOCHS,
        bf16 = True,
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
            mlm_probability = 0.15
        ),
        train_dataset = dataset_train_tokenized,
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
    dataset_train_tokenized = load_dataset(tokenizer)

    # 加载模型
    model = load_model()

    # 打印模型的参数量
    print_model_parameters(model)

    # 设置 wandb
    wandb.init(
        project = "PRETRAIN",
        name = f"{MODEL_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}",
    )

    # 开始训练
    start_training(model, tokenizer, dataset_train_tokenized)

# 主函数
if __name__ == "__main__":
    main()