import os
import re
import random
import shutil
import unicodedata
from typing import Generator
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
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers.utils import is_torch_bf16_gpu_available
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from callback.PreTrainerCallback import PreTrainerCallback


# 模型
SCRATCH = True
MODEL_NAME = "modern_bert_cjk"
MODEL_PATH = f"assets/{MODEL_NAME}"
OUTPUT_PATH = f"output/{MODEL_NAME}_pt_e1"
ATTN_IMPLEMENTATION = "flash_attention_2" # sdpa, flex_attention, flash_attention_2, eager

# 训练
SEED = 42
WEIGHT_DECAY = 1 * 1e-5
LEARNING_RATE = 5 * 1e-4
EPOCHS = 1
EVAL_SIZE = 8
BATCH_SIZE = 24
TORCH_COMPILE = True
GRADIENT_CHECKPOINTING = True
GRADIENT_ACCUMULATION_SIZE = 256

# 输出
LOG_STEPS = 5
INTERVAL_STEPS = 300
AUTO_RESUME_FROM_CHECKPOINT = True

# 数据
EVAL_DATA = 8
LENGTH_THRESHOLD = 256
DATASET_PATH = [
    # ("dataset/pt/zh", 20 * 10000),
    # ("dataset/pt/zh_r18_pixiv", 20 * 10000),
    # ("dataset/pt/en", 30 * 10000),
    # ("dataset/pt/en_r18_visual_novels", 10 * 10000),
    # ("dataset/pt/jp", 40 * 10000),
    # ("dataset/pt/jp_r18", 32.5 * 10000),
    # ("dataset/pt/jp_r18_rpg", 7.5 * 10000),
    # ("dataset/pt/ko", 20 * 10000),
    # ("dataset/pt/ko_web", 20 * 10000),
    ("dataset/pt/zh_cc100", 800 * 10000),
    ("dataset/pt/zh_cc100_tw", 800 * 10000),
    ("dataset/pt/en_c4", 800 * 10000),
    ("dataset/pt/jp_cc100", 800 * 10000),
    ("dataset/pt/ko_cc100", 800 * 10000),
]

# 加载模型
def load_model(scratch: bool) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.reference_compile = None

    if scratch == True:
        return AutoModelForMaskedLM.from_config(
            config,
            attn_implementation = ATTN_IMPLEMENTATION,
            torch_dtype = torch.bfloat16 if is_torch_bf16_gpu_available() == True else torch.float16,
            trust_remote_code = True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return AutoModelForMaskedLM.from_pretrained(
            MODEL_PATH,
            config = config,
            attn_implementation = ATTN_IMPLEMENTATION,
            torch_dtype = torch.bfloat16 if is_torch_bf16_gpu_available() == True else torch.float16,
            local_files_only = True,
            trust_remote_code = True,
            ignore_mismatched_sizes = True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

# 加载分词器
def load_tokenizer() -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(
        MODEL_PATH,
        do_lower_case = False,
        local_files_only = True,
    )

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

# 分割列表
def split_list(lst: list, batch_size: int) -> list[list]:
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

# 批量读取文本文件
def lines_generator(path: str, batch_size: int, flag: dict) -> Generator[list, any, Dataset]:
    # 初始化文件列表
    files = [file for file in os.scandir(path) if file.name.endswith(".txt")]
    files = random.sample(files, len(files))

    # 开始生成数据
    lines = []
    for file in tqdm(files, desc = path, total = len(files)):
        # 根据信号判断是否需要停止生成数据
        if flag.get("stop", False) == True:
            break

        with open(file.path, "r", encoding = "utf-8") as reader:
            for line in reader:
                # 根据信号判断是否需要停止生成数据
                if flag.get("stop", False) == True:
                    break

                line = line.strip()

                if line != "":
                    lines.append(line)

                if len(lines) >= batch_size:
                    yield lines
                    lines = []

            if len(lines) > 0:
                yield lines
                lines = []

# 生成数据
def datas_generator(tokenizer: PreTrainedTokenizerFast, lines: list[str], path: str) -> list[str]:
    lines = [cleanup(line, path) for line in lines]

    datas = []
    encodings = tokenizer(
        lines,
        padding = False,
        truncation = True,
        max_length = LENGTH_THRESHOLD + 8, # 加一些冗余，避免加上特殊 Token 以后长度刚刚好的情况
        return_special_tokens_mask = True,
    )

    # 计算文本的长度，此处只统计实际有效的 Token 数量
    datas = [
        {
            "line": line,
            "length": special_tokens_mask.count(0),
        }
        for line, special_tokens_mask in zip(lines, encodings.get("special_tokens_mask"))
    ]

    return datas

# 生成数据块
def chunks_generator(tokenizer: PreTrainedTokenizerFast, lines: list[str], path: str) -> list[list[str]]:
    chunks = []
    datas = datas_generator(tokenizer, lines, path)

    chunk = ""
    chunk_length = 0
    for data in datas:
        line = data.get("line")
        length = data.get("length")

        # 计算片段的长度，如果会超过阈值则分割
        # Tokenizer 一般会在首尾各加入一个特殊 Token，所以此处预留 2 个冗余位置
        if chunk_length + length >= LENGTH_THRESHOLD - 2:
            # 去除重复空格并存储当前chunk
            chunks.append(re.sub(r" +", " ", f"{chunk} {line}").strip())

            # 重置片段
            chunk = ""
            chunk_length = 0
        else:
            # 如果当前片段未超过阈值，继续拼接
            chunk = chunk + " " + line

            # 空格算不算 Token 都有可能，保险起见按计入计算，即长度 +1
            chunk_length = chunk_length + 1 + length

     # 最后一个片段如果非空，则添加
    if chunk.strip() != "":
        chunks.append(re.sub(r" +", " ", f"{chunk} {line}").strip())

    return chunks

# 准备语料
def generate_text_file(path: str, output: str, tokenizer: PreTrainedTokenizerFast, threshold: int) -> None:
    # 初始化控制信号
    flag = {
        "stop": False,
    }

    # 并行处理数据分段进行
    data = []
    with Parallel(n_jobs = os.cpu_count(), prefer = "processes", return_as = "generator_unordered") as parallel:
        results = parallel(
            delayed(chunks_generator)(tokenizer, lines, path) for lines in lines_generator(path, 32 * 1024, flag)
        )
        for result in results:
            data.extend(result)
            flag["current"] = len(data)
            if len(data) >= threshold:
                flag["stop"] = True

    # 按阈值随机取数据
    if threshold <= len(data):
        data = random.sample(data, int(threshold))
    else:
        print(f"{path}: 数据量不足，将重复数据以满足需求，{len(data)} -> {threshold} ...")
        data = data + random.sample(data, int(threshold - len(data)))

    # 按阈值随机取数据，然后写入文件
    data = random.sample(data, int(threshold))
    with open(output, "w", encoding = "utf-8") as writer:
        writer.writelines(tqdm((f"{line}\n" for line in data), desc = path, total = len(data)))

# 加载数据集
def load_dataset(tokenizer: PreTrainedTokenizerFast) -> tuple[Dataset, Dataset]:
    print("")
    print("正在加载数据集 ...")
    print("")

    # 加载或者生成数据集
    cache_path = "dataset/pt/cache"
    os.makedirs(cache_path, exist_ok = True)
    if not os.path.isdir(f"{cache_path}/{MODEL_NAME}_tokenized"):
        # 遍历数据集路径
        paths = []
        for path, threshold in DATASET_PATH:
            dir_path, dir_name = os.path.split(path)

            # 如果数据文本文件不存在，则生成
            output = f"{dir_path}/{MODEL_NAME}_{dir_name}.txt"
            if os.path.isfile(output) == True:
                pass
            else:
                generate_text_file(path, output, tokenizer, threshold)

            # 记录路径
            paths.append(output)

        dataset_tokenized = Dataset.from_text(
            paths,
            cache_dir = cache_path
        ).map(
            lambda samples: load_dataset_map_function(samples, tokenizer),
            num_proc = os.cpu_count(),
            batched = True,
            remove_columns = ["text"],
            cache_file_name = f"{cache_path}/map/{MODEL_NAME}.cache",
            load_from_cache_file = True,
        )
        dataset_tokenized.save_to_disk(
            dataset_path = f"{cache_path}/{MODEL_NAME}_tokenized",
            num_proc = os.cpu_count(),
            max_shard_size = "4GB",
        )
        shutil.rmtree(f"{cache_path}/map", ignore_errors = True)
        shutil.rmtree(f"{cache_path}/text", ignore_errors = True)
        [os.remove(file.path) for file in os.scandir(f"{cache_path}") if file.name.endswith(".lock")]
    dataset_tokenized = Dataset.load_from_disk(f"{cache_path}/{MODEL_NAME}_tokenized")

    # 统计数据
    max_length = max(dataset_tokenized["attention_length"])
    total_length = sum(dataset_tokenized["attention_length"])

    # 拆分数据集
    [os.remove(file.path) for file in os.scandir(f"{cache_path}") if file.name.endswith("_indices.cache")]
    dataset_dict = dataset_tokenized.train_test_split(
        seed = SEED,
        shuffle = True,
        test_size = EVAL_DATA,
        test_indices_cache_file_name = f"{cache_path}/{MODEL_NAME}_eval_indices.cache",
        train_indices_cache_file_name = f"{cache_path}/{MODEL_NAME}_train_indices.cache",
    )
    eval_dataset, train_dataset = dataset_dict.get("test"), dataset_dict.get("train")

    print("")
    print("数据加载已完成 ... 样本如下：")
    print("")
    print_dataset_sample(tokenizer, dataset_tokenized)
    print("")
    print(""
        + f"共加载 {len(dataset_tokenized)} 条数据，其中有效 Token {(total_length / 1000 / 1000):.2f} M，"
        + f"最长条目 {(max_length):.2f} Token，平均每个条目 {(total_length / len(dataset_tokenized)):.2f} Token ..."
    )

    return eval_dataset, train_dataset

# 映射函数
def load_dataset_map_function(samples: dict, tokenizer: PreTrainedTokenizerFast) -> BatchEncoding:
    encodings = tokenizer(
        samples.get("text"),
        padding = "max_length",
        truncation = True,
        max_length = LENGTH_THRESHOLD, # 最大长度是包含特殊 ID 在内的，所以不需要增减
        return_attention_mask = True,
        return_special_tokens_mask = True,
    )

    # 生成 input_tokens
    # encodings["input_tokens"] = []
    # for tokens in [tokenizer.convert_ids_to_tokens(v) for v in encodings.get("input_ids")]:
    #     encodings["input_tokens"].append([])
    #     for t in tokens:
    #         encodings["input_tokens"][-1].append(tokenizer.convert_tokens_to_string([t]))

    # 计算有效的 Token 数量
    encodings["attention_length"] = [item.count(1) for item in encodings.get("attention_mask")]

    return encodings

# 打印数据集样本
def print_dataset_sample(tokenizer: PreTrainedTokenizerFast, dateset: Dataset) -> None:
    if len(dateset) == 0:
        return

    input_ids = dateset[0].get("input_ids")
    input_tokens = tokenizer.batch_decode(input_ids)
    attention_mask = dateset[0].get("attention_mask")
    special_tokens_mask = dateset[0].get("special_tokens_mask")

    print(f"{"tokens":<8}\t\t{"ids":<4}\t\t{"attention":<8}\t\t{"special_mask":<6}")
    for x, z, a, b in zip(input_tokens, input_ids, attention_mask, special_tokens_mask):
        print(f"{x:<8}\t\t{z:<4}\t\t{a:<8}\t\t{b:<6}")

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
    # Graph break from `Tensor.item()`, consider setting:
    # torch._dynamo.config.capture_scalar_outputs = True or env TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
    # to include these operations in the captured graph.
    # if TORCH_COMPILE == True:
    #     torch._dynamo.config.capture_scalar_outputs = True

    training_args = TrainingArguments(
        # 输出
        report_to = "wandb",
        output_dir = OUTPUT_PATH,
        logging_steps = LOG_STEPS,
        eval_steps = INTERVAL_STEPS,
        save_steps = INTERVAL_STEPS,
        eval_strategy = "steps",
        save_strategy = "no",

        # 训练
        bf16 = True,
        torch_compile = TORCH_COMPILE,
        # optim = "adamw_torch",
        optim = "paged_adamw_32bit",
        # optim = "paged_ademamix_8bit",
        adam_beta1 = 0.90,
        adam_beta2 = 0.98,
        adam_epsilon = 1e-06,
        warmup_ratio = 0.1,
        weight_decay = WEIGHT_DECAY,
        learning_rate = LEARNING_RATE,
        num_train_epochs = EPOCHS,
        lr_scheduler_type = "warmup_stable_decay",
        lr_scheduler_kwargs = {
            "num_decay_steps": int(len(train_dataset) * 0.1 / max(BATCH_SIZE, GRADIENT_ACCUMULATION_SIZE)) + 1,
            "num_stable_steps": int(len(train_dataset) * 0.8 / max(BATCH_SIZE, GRADIENT_ACCUMULATION_SIZE)) + 1,
        },
        per_device_eval_batch_size = EVAL_SIZE,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_checkpointing = GRADIENT_CHECKPOINTING,
        gradient_accumulation_steps = int(max(BATCH_SIZE, GRADIENT_ACCUMULATION_SIZE) / BATCH_SIZE),
    )

    trainer = Trainer(
        args = training_args,
        model = model,
        data_collator = DataCollatorForLanguageModeling(
            tokenizer = tokenizer,
            mlm = True,
            mlm_probability = 0.30,
            pad_to_multiple_of = 8,
        ),
        eval_dataset = eval_dataset,
        train_dataset = train_dataset,
        processing_class = tokenizer,
    )
    trainer.add_callback(
        PreTrainerCallback(
            trainer = trainer,
        ),
    )

    # 检查是否自动恢复训练
    resume_from_checkpoint = f"{OUTPUT_PATH}/latest" if AUTO_RESUME_FROM_CHECKPOINT == True and os.path.isdir(f"{OUTPUT_PATH}/latest") else None
    if resume_from_checkpoint != None:
        print(f"在 {OUTPUT_PATH} 找到可恢复的训练状态，自动继续训练 ...")
    trainer.train(
        resume_from_checkpoint = resume_from_checkpoint,
    )

# 主函数
def main() -> None:
    # 固定随机种子
    random.seed(SEED)

    # 加载分词器
    tokenizer = load_tokenizer()

    # 加载数据集
    eval_dataset, train_dataset = load_dataset(tokenizer)

    # 加载模型
    model = load_model(SCRATCH)

    # 调整 token_embeddings 的大小
    if SCRATCH == False:
        model.resize_token_embeddings(len(tokenizer))

    # 打印模型的参数量
    print_model_parameters(model)

    # 设置 wandb
    wandb.init(
        project = "PT",
        name = f"{MODEL_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}",
    )

    # 开始训练
    start_training(model, tokenizer, eval_dataset, train_dataset)

# 主函数
if __name__ == "__main__":
    main()