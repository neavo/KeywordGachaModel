import os
import random
import shutil
from typing import Generator
from datetime import datetime

import torch
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

from moudle.Normalizer import Normalizer
from callback.MemoryCallback import MemoryCallback
from callback.PreTrainerCallback import PreTrainerCallback

# 模型
SCRATCH = True
MODEL_NAME = "modern_bert_cjk"
MODEL_PATH = f"assets/{MODEL_NAME}"
OUTPUT_PATH = f"output/{MODEL_NAME}_pt_e1"
ATTN_IMPLEMENTATION = "sdpa" # sdpa, flex_attention, flash_attention_2, eager

# 训练
SEED = 42
WEIGHT_DECAY = 1 * 1e-5
LEARNING_RATE = 5 * 1e-4
EPOCHS = 1
OPTIMIZER = "adamw_torch" # adamw_torch, adamw_torch_fused, paged_adamw_8bit, paged_lion_8bit, paged_ademamix_8bit
EVAL_SIZE = 16
BATCH_SIZE = 8
PRECISION = "bf16" # bf16, fp16, bf16_pure
TORCH_COMPILE = True
GRADIENT_CHECKPOINTING = True
GRADIENT_ACCUMULATION_SIZE = 256

# 输出
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 5
AUTO_RESUME_FROM_CHECKPOINT = True
WANDB_ENABLE = True

# 数据
EVAL_DATA = 2048 * 8
LENGTH_THRESHOLD = 512
WORKSPACE = "workspace"
DATASET_PATH = [
    # ("/mnt/e/ai/dataset/pt/zh", 40 * 10000),
    # ("/mnt/e/ai/dataset/pt/zh_r18_pixiv", 40 * 10000),
    # ("/mnt/e/ai/dataset/pt/en", 60 * 10000),
    # ("/mnt/e/ai/dataset/pt/en_r18_visual_novels", 20 * 10000),
    # ("/mnt/e/ai/dataset/pt/ja", 80 * 10000),
    # ("/mnt/e/ai/dataset/pt/ja_r18", 65 * 10000),
    # ("/mnt/e/ai/dataset/pt/ja_r18_rpg", 15 * 10000),
    # ("/mnt/e/ai/dataset/pt/ko", 40 * 10000),
    # ("/mnt/e/ai/dataset/pt/ko_web", 40 * 10000),
    ("/mnt/e/ai/dataset/pt/zh_cc100", 800 * 10000),
    ("/mnt/e/ai/dataset/pt/zh_cc100_tw", 400 * 10000),
    ("/mnt/e/ai/dataset/pt/en_cc100", 800 * 10000),
    ("/mnt/e/ai/dataset/pt/ja_cc100_izumi_lab", 1200 * 10000),
    ("/mnt/e/ai/dataset/pt/ko_cc100", 800 * 10000),
]

# 加载模型
def load_model() -> PreTrainedModel:
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.reference_compile = None

    if SCRATCH == True:
        return AutoModelForMaskedLM.from_config(
            config,
            attn_implementation = ATTN_IMPLEMENTATION,
            torch_dtype = torch.bfloat16 if PRECISION == "bf16_pure" and is_torch_bf16_gpu_available() == True else None,
            trust_remote_code = True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return AutoModelForMaskedLM.from_pretrained(
            MODEL_PATH,
            config = config,
            attn_implementation = ATTN_IMPLEMENTATION,
            torch_dtype = torch.bfloat16 if PRECISION == "bf16_pure" and is_torch_bf16_gpu_available() == True else None,
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

# 分割列表
def split_list(lst: list, batch_size: int) -> list[list]:
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

# 批量读取文本文件
def lines_generator(path: str, batch_size: int, flag: dict) -> Generator[list[str], None, None]:
    # 初始化文件列表
    files = [file for file in os.scandir(path) if file.name.endswith(".txt")]
    files = random.sample(files, len(files))

    # 开始生成数据
    lines: list[str] = []
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
def datas_generator(tokenizer: PreTrainedTokenizerFast, lines: list[str]) -> list[str]:
    lines = [Normalizer.normalize(line, merge_space = True) for line in lines]

    # 获取特殊 Token 的数量
    special_tokens_num = tokenizer.num_special_tokens_to_add(pair = False)

    # 计算文本的编码
    encodings = tokenizer(
        lines,
        padding = False,
        truncation = True,
        max_length = LENGTH_THRESHOLD + special_tokens_num, # 加上冗余长度，加上特殊 Token 以后长度刚刚好的情况
    )

    # 计算文本的长度，此处只统计实际有效的 Token 数量
    datas = [
        {
            "line": line,
            "length": len(input_ids) - special_tokens_num,
        }
        for line, input_ids in zip(lines, encodings.get("input_ids"))
    ]

    return datas

# 生成数据块
def chunks_generator(tokenizer: PreTrainedTokenizerFast, lines: list[str], path: str) -> list[list[str]]:
    result: list[str] = []
    datas = datas_generator(tokenizer, lines)

    # 获取特殊 Token 的数量
    special_tokens_num = tokenizer.num_special_tokens_to_add(pair = False)

    # 遍历数据，将数据分段
    chunk: str = ""
    chunk_length = 0
    for data in datas:
        line = data.get("line")
        length = data.get("length")

        # 拼接片段
        if len(chunk) == 0:
            chunk = line
            chunk_length = length
        else:
            chunk = f"{chunk} {line}"
            chunk_length = chunk_length + length + 1

        # 计算片段的 Token 长度，如果超过阈值，则添加到结果中并开始新的分段
        # Tokenizer 会在收尾加入特殊 Token，文本之间还会有一个连接符，所以要减去这些长度
        if chunk_length >= LENGTH_THRESHOLD - special_tokens_num - 1:
            result.append(chunk)
            chunk = ""
            chunk_length = 0

     # 最后一个片段如果非空且有一定长度，则添加到结果
    if chunk.strip() != "" and chunk_length > (LENGTH_THRESHOLD - special_tokens_num - 1) * 0.8:
        result.append(chunk)

    return result

# 准备语料
def generate_text_file(path: str, output: str, tokenizer: PreTrainedTokenizerFast, threshold: int) -> None:
    # 初始化控制信号
    flag = {
        "stop": False,
    }

    # 并行处理数据分段进行
    with Parallel(n_jobs = os.cpu_count(), prefer = "processes", return_as = "generator_unordered") as parallel:
        results = parallel(
            delayed(chunks_generator)(tokenizer, lines, path) for lines in lines_generator(path, 32 * 1000, flag)
        )

        # 处理结果
        data = []
        for result in results:
            data.extend(result)
            flag["stop"] = len(data) >= threshold

    # 按阈值随机取数据
    if threshold <= len(data):
        data = random.sample(data, int(threshold))
    else:
        print(f"{path}: 数据量不足，将重复数据以满足需求，{len(data)} -> {int(threshold)} ...")
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

    # 如果数据集不存在，则生成数据集
    os.makedirs(f"{WORKSPACE}/cache", exist_ok = True)
    if not os.path.isdir(f"{WORKSPACE}/{MODEL_NAME}_tokenized"):
        # 遍历数据集路径
        paths = []
        for path, threshold in DATASET_PATH:
            _, dir_name = os.path.split(path)

            # 如果数据文本文件不存在，则生成
            output = f"{WORKSPACE}/{MODEL_NAME}_{dir_name}.txt"
            if os.path.isfile(output) == False:
                generate_text_file(path, output, tokenizer, threshold)

            # 记录路径
            paths.append(output)

        dataset_tokenized = Dataset.from_text(
            paths,
            cache_dir = f"{WORKSPACE}/cache"
        ).map(
            lambda samples: load_dataset_map_function(samples, tokenizer),
            num_proc = os.cpu_count(),
            batched = True,
            remove_columns = ["text"],
            cache_file_name = f"{WORKSPACE}/cache/map/{MODEL_NAME}.cache",
            load_from_cache_file = True,
        )
        dataset_tokenized.save_to_disk(
            dataset_path = f"{WORKSPACE}/{MODEL_NAME}_tokenized",
            num_proc = os.cpu_count(),
            max_shard_size = "4GB",
        )

    # 清理缓存并加载数据集
    shutil.rmtree(f"{WORKSPACE}/cache", ignore_errors = True)
    dataset_tokenized = Dataset.load_from_disk(f"{WORKSPACE}/{MODEL_NAME}_tokenized")

    # 统计数据
    max_length = max(dataset_tokenized["length"])
    total_length = sum(dataset_tokenized["length"])

    # 拆分数据集
    dataset_dict = dataset_tokenized.train_test_split(
        seed = SEED,
        shuffle = True,
        test_size = EVAL_DATA,
        test_indices_cache_file_name = f"{WORKSPACE}/cache/{MODEL_NAME}_eval_indices.cache",
        train_indices_cache_file_name = f"{WORKSPACE}/cache/{MODEL_NAME}_train_indices.cache",
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

    # 计算有效的 Token 数量
    encodings["length"] = [item.count(0) for item in encodings.get("special_tokens_mask")]

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
        report_to = "wandb" if WANDB_ENABLE == True else "none",
        output_dir = OUTPUT_PATH,
        eval_steps = EVAL_STEPS,
        save_steps = SAVE_STEPS,
        logging_steps = LOGGING_STEPS,
        eval_strategy = "steps" if EVAL_STEPS != None and EVAL_STEPS > 0 else "no",
        save_strategy = "steps" if SAVE_STEPS != None and SAVE_STEPS > 0 else "no",
        logging_strategy = "steps" if LOGGING_STEPS != None and LOGGING_STEPS > 0 else "no",

        # 训练
        torch_compile = TORCH_COMPILE,
        bf16 = PRECISION in ("bf16", "bf16_pure"),
        optim = OPTIMIZER,
        warmup_ratio = 0.10,
        weight_decay = WEIGHT_DECAY,
        learning_rate = LEARNING_RATE,
        num_train_epochs = EPOCHS,
        lr_scheduler_type = "warmup_stable_decay",
        lr_scheduler_kwargs = {
            "num_decay_steps": int(len(train_dataset) * EPOCHS * 0.10 / max(BATCH_SIZE, GRADIENT_ACCUMULATION_SIZE)) + 1,
            "num_stable_steps": int(len(train_dataset) * EPOCHS * 0.80 / max(BATCH_SIZE, GRADIENT_ACCUMULATION_SIZE)) + 1,
        },
        per_device_eval_batch_size = EVAL_SIZE,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_checkpointing = GRADIENT_CHECKPOINTING,
        gradient_accumulation_steps = int(max(BATCH_SIZE, GRADIENT_ACCUMULATION_SIZE) / BATCH_SIZE),
        dataloader_pin_memory = True,
        dataloader_num_workers = 8,
        dataloader_persistent_workers = True,
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
        )
    )
    # trainer.add_callback(
    #     MemoryCallback(
    #         threshold = 0.93,
    #         check_steps = 1,
    #         force_clean_on_start = True,
    #     )
    # )

    # 检查是否自动恢复训练
    resume_from_checkpoint = f"{OUTPUT_PATH}/latest" if AUTO_RESUME_FROM_CHECKPOINT == True and os.path.isdir(f"{OUTPUT_PATH}/latest") else None
    if resume_from_checkpoint != None:
        print("")
        print(f"在 [green]{OUTPUT_PATH}[/] 找到可恢复的训练状态，自动继续训练 ...")
        print("")
    trainer.train(
        resume_from_checkpoint = resume_from_checkpoint,
    )
    trainer.accelerator.free_memory()

# 主函数
def main() -> None:
    # 固定随机种子
    random.seed(SEED)

    # 加载分词器
    tokenizer = load_tokenizer()

    # 加载数据集
    eval_dataset, train_dataset = load_dataset(tokenizer)

    # 加载模型
    model = load_model()

    # 调整 token_embeddings 的大小
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # 打印模型的参数量
    print_model_parameters(model)

    # 设置 wandb
    if WANDB_ENABLE == True:
        import wandb
        wandb.init(
            project = "PT",
            name = f"{MODEL_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}",
        )

    # 开始训练
    start_training(model, tokenizer, eval_dataset, train_dataset)

    # 结束 wandb
    if WANDB_ENABLE == True:
        wandb.finish()

    # 退出
    os._exit(0)

# 主函数
if __name__ == "__main__":
    main()