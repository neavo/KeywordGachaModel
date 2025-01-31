import os
import re
import json
import random
from datetime import datetime

from rich import print

import numpy
import torch
from datasets import Dataset
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import EvalPrediction
from transformers import PreTrainedModel
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from seqeval.metrics import f1_score
from seqeval.metrics import recall_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import precision_score
from seqeval.metrics import classification_report

from callback.MemoryCallback import MemoryCallback
from callback.NERTrainerCallback import NERTrainerCallback

# 任务
START_DATE = datetime.now().strftime("%Y%m%d")
START_TIME = datetime.now().strftime("%H%M%S")
WANDB_ENABLE = True

# 模型
INPUT_NAME = "keyword_gacha_multilingual"
INPUT_PATH = f"assets/{INPUT_NAME}/20250128/latest"
OUTPUT_NAME = "keyword_gacha_multilingual_ner"
OUTPUT_PATH = f"output/{OUTPUT_NAME}/{START_DATE}_5e5_cosine"
PROJECT_NAME = f"{OUTPUT_NAME}_{START_DATE}_5e5_cosine"
ATTN_IMPLEMENTATION = "flash_attention_2" # sdpa, flash_attention_2, eager

# 训练
SEED = 42
PATIENCE = 999
OPTIMIZER = "adamw_torch" # adamw_torch, adamw_torch_fused, paged_adamw_8bit, paged_lion_8bit, paged_ademamix_8bit
MAX_STEPS = 7500
EVAL_SIZE = 128
BATCH_SIZE = 32
TORCH_COMPILE = False
FROZEN_LAYER = 0
WEIGHT_DECAY = 1 * 1e-5
LEARNING_RATE = 5 * 1e-5
GRADIENT_CHECKPOINTING = False
GRADIENT_ACCUMULATION_SIZE = 0

# 输出
SAVE_STEPS = 0
EVAL_STEPS = 200
LOGGING_STEPS = 5
WARMUP_STEPS = 750

# 数据
EVAL_DATA = 4096
DATASET_PATH = [
    # ("/mnt/e/ai/dataset/ner/zh/20250102", 1.5 * 10000 + EVAL_DATA / 8),
    # ("/mnt/e/ai/dataset/ner/en/20250102", 1.5 * 10000 + EVAL_DATA / 8),
    # ("/mnt/e/ai/dataset/ner/ja/20250102", 1.5 * 10000 + EVAL_DATA / 8),
    # ("/mnt/e/ai/dataset/ner/ko/20250102", 1.5 * 10000 + EVAL_DATA / 8),
    ("/mnt/e/ai/dataset/ner/zh/20250121", 2.5 * 10000 + EVAL_DATA / 4),
    ("/mnt/e/ai/dataset/ner/en/20250121", 2.5 * 10000 + EVAL_DATA / 4),
    ("/mnt/e/ai/dataset/ner/ja/20250121", 2.5 * 10000 + EVAL_DATA / 4),
    ("/mnt/e/ai/dataset/ner/ko/20250121", 2.5 * 10000 + EVAL_DATA / 4),
]

# 加载模型
def load_model(id2label: dict, label2id: dict) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(
        INPUT_PATH,
        local_files_only = True,
        trust_remote_code = True,
    )
    config.id2label = id2label
    config.label2id = label2id
    config.num_labels = len(id2label)

    return AutoModelForTokenClassification.from_pretrained(
        INPUT_PATH,
        config = config,
        local_files_only = True,
        trust_remote_code = True,
        ignore_mismatched_sizes = True,
        attn_implementation = ATTN_IMPLEMENTATION,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

# 加载分词器
def load_tokenizer() -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(
        INPUT_PATH,
        do_lower_case = False,
        local_files_only = True,
    )

# 随机取样，如果数量充足，则尽可能平衡类型
def sample(data: list[dict], limit: int) -> list[dict]:
    # 找出最大的类型
    type_count = {}
    for item in data:
        for entity in item.get("entities", []):
            type_count[entity.get("entity_group")] = type_count.get(entity.get("entity_group"), 0) + 1
    max_k = max(type_count, key = lambda k: type_count.get(k), default="")

    # 拆分数据
    data_x = [item for item in data if any(entity.get("entity_group") != max_k for entity in item.get("entities", []))]
    data_y = [item for item in data if not any(entity.get("entity_group") != max_k for entity in item.get("entities", []))]

    # 随机取样
    if len(data_x) >= limit:
        return random.sample(data_x, limit)
    else:
        return random.sample(data_x, len(data_x)) + random.sample(data_y, min(len(data_y), limit - len(data_x)))

# 加载数据集
def load_dataset(tokenizer: PreTrainedTokenizerFast) -> tuple[Dataset, Dataset, dict, dict]:
    data = []
    count = 0
    for path, limit in DATASET_PATH:
        if os.path.isfile(path) == True:
            data_ex = []
            with open(path, "r", encoding = "utf-8") as file:
                count = count + 1
                data_ex.extend(json.load(file))
            data.extend(sample(data_ex, int(limit)))
        elif os.path.isdir(path) == True:
            data_ex = []
            for file in [file for file in os.scandir(path) if file.path.endswith(".json")]:
                with open(file.path, "r", encoding = "utf-8") as file:
                    count = count + 1
                    data_ex.extend(json.load(file))
            data.extend(sample(data_ex, int(limit)))

    # 只取需要的字段，避免后续转换格式时的错误
    data = [{"sentence": v.get("sentence", ""), "entities": v.get("entities", [])} for v in data]

    # 生成 id-label 映射表
    types = set()
    for v in data:
        for entity in v.get("entities", []):
            if entity.get("entity_group") != "":
                types.add(entity.get("entity_group"))
    id2label = {0: "O"}
    for c in list(sorted(types)):
        id2label[len(id2label)] = f"B-{c}"
        id2label[len(id2label)] = f"I-{c}"
    label2id = {v: k for k, v in id2label.items()}

    # 生成数据集
    dataset_tokenized = Dataset.from_list(data).map(
        lambda samples: load_dataset_map_function(samples, tokenizer, label2id),
        batched = True,
        remove_columns = ["sentence", "entities"],
        keep_in_memory = True
    )

    # 获取最大长度
    max_length = max(len(v.get("input_ids")) for v in dataset_tokenized)

    # 拆分数据集
    dataset_dict = dataset_tokenized.train_test_split(
        seed = SEED,
        shuffle = True,
        test_size = EVAL_DATA,
    )
    eval_dataset, train_dataset = dataset_dict.get("test"), dataset_dict.get("train")

    print("")
    print("数据加载已完成 ... 样本如下：")
    print("")
    print_dataset_sample(tokenizer, dataset_tokenized, id2label)
    print("")
    print(f"加载数据文件 {count} 个，共 {len(data)} 条数据，最大长度为 {max_length} ...")

    return eval_dataset, train_dataset, id2label, label2id, max_length

# 打印数据集样本
def print_dataset_sample(tokenizer: PreTrainedTokenizerFast, dateset: Dataset, id2label: dict) -> None:
    if len(dateset) == 0:
        return

    labels = dateset[0].get("labels")
    input_ids = dateset[0].get("input_ids")
    input_tokens = tokenizer.batch_decode(input_ids)
    attention_mask = dateset[0].get("attention_mask")
    special_tokens_mask = dateset[0].get("special_tokens_mask")

    print(f"{"tokens":<8}\t\t{"labels":<4}\t\t{"ids":<4}\t\t{"attention":<8}\t\t{"special_mask":<6}")
    for x, y, z, a, b in zip(input_tokens, labels, input_ids, attention_mask, special_tokens_mask):
        print(f"{x:<8}\t\t{id2label.get(y):<4}\t\t{z:<4}\t\t{a:<8}\t\t{b:<6}")

# 数据集映射函数
def load_dataset_map_function(samples: dict, tokenizer: PreTrainedTokenizerFast, label2id: dict) -> BatchEncoding:
    encodings = tokenizer(
        samples.get("sentence"),
        return_attention_mask = True,
        return_offsets_mapping = True,
        return_special_tokens_mask = True,
    )

    # 生成 labels
    for i, _ in enumerate(encodings.get("input_ids")):
        sentence = samples.get("sentence", [])[i]
        entities = samples.get("entities", [])[i]
        input_ids = encodings.get("input_ids")[i]
        offsets_mapping = encodings.get("offset_mapping")[i]

        # 遍历实体词语
        result = []
        for entity in entities:
            surface = entity.get("surface", "")
            entity_group = entity.get("entity_group", "")

            # 获取实体词语在字符串中的位置
            char_start = sentence.find(surface)
            char_end = char_start + len(surface)

            # 有效性检查
            if char_start < 0 or surface == "" or entity_group == "":
                continue

            # 通过字符位置反查 Token 位置
            token_start, token_end = char_offset_to_token_offset(char_start, char_end, offsets_mapping)

            # 跳过不存在的 Token
            if token_start == -1 or token_end == -1:
                continue

            result.append((token_start, token_end, entity_group))

        # 生成 labels
        labels = [0 for _ in range(len(input_ids))]
        for i in range(len(input_ids)):
            for v in result:
                if v[0] == i:
                    labels[i] = label2id.get(f"B-{v[2]}", 0)
                elif v[0] < i < v[1]:
                    labels[i] = label2id.get(f"I-{v[2]}", 0)

        # 添加 labels
        encodings.setdefault("labels", []).append(labels)

    return encodings

# 通过字符位置反查 token 位置
def char_offset_to_token_offset(char_start, char_end, offsets_mapping) -> tuple[int, int]:
    token_end = -1
    token_start = -1

    for i, (start, end) in enumerate(offsets_mapping):
        # 起始位置一致的 Token 在字符串中实际上不存在，跳过它
        if start == end:
            continue

        # 当前 Token 不是最后一个 Token，且与下一个 Token 的起始位置一致
        # 则可能是 SentencePiece 向句子开头添加内容为 _ 的 token，跳过它
        if i < len(offsets_mapping) - 1 and offsets_mapping[i][0] == offsets_mapping[i + 1][0]:
            continue

        if start <= char_end < end:
            token_end = i
            break

    for i, (start, end) in enumerate(offsets_mapping):
        # 起始位置一致的 Token 在字符串中实际上不存在，跳过它
        if start == end:
            continue

        # 当前 Token 不是最后一个 Token，且与下一个 Token 的起始位置一致
        # 则可能是 SentencePiece 向句子开头添加内容为 _ 的 token，跳过它
        if i < len(offsets_mapping) - 1 and offsets_mapping[i][0] == offsets_mapping[i + 1][0]:
            continue

        if start <= char_end < end:
            token_end = i
            break

        if start <= char_start < end:
            token_start = i
            break

    return token_start, token_end

# 设置模型层
def set_layers(model: PreTrainedModel) -> None:
    # 微调时，我们通常冻结除了最后几层以外的所有层
    # 低层（Lower layers）：靠近输入层，标号较小的层。例如，第1层、第2层等。
    # 高层（Higher layers）：靠近输出层，标号较大的层。例如，第11层、第12层等。
    # 因此，当我们说冻结低层时，指的是冻结这些靠近输入端的层，而仅训练靠近输出端的高层。
    # 这是因为低层通常捕捉到的是更通用的语言特征，而高层则更多地关注任务特定的特征。
    for name, param in model.named_parameters():
        layer_num = re.findall(r"\d+", name)
        if "encoder.layer" in name and len(layer_num) > 0 and int(layer_num[0]) + 1 <= FROZEN_LAYER:
            param.requires_grad = False
            print(f"已冻结 - {name} ...")

# 打印模型的参数量，按 M 格式化
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
    print(f"{INPUT_NAME} : layer - {layer / 1e6:.2f} M / embedding - {embedding / 1e6:.2f} M / total - {total / 1e6:.2f} M")
    print("")

# 计算评估指标
def compute_metrics(eval_prediction: EvalPrediction, id2label: dict) -> dict:
    predictions, labels = eval_prediction
    predictions = numpy.argmax(predictions, axis = 2) # 对于 3 维张量， axis = 2 与 axis = -1 是一样的

    true_labels = [
        [id2label[l] for p, l in zip(pred, lab) if p != -100 and l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    true_predictions = [
        [id2label[p] for p, l in zip(pred, lab) if p != -100 and l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    return {
        "f1": f1_score(true_labels, true_predictions, average = "micro", zero_division = 0),
        "recall": recall_score(true_labels, true_predictions, average = "micro", zero_division = 0),
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions, average = "micro", zero_division = 0),
        "classification_report": classification_report(true_labels, true_predictions, output_dict = True, zero_division = 0)
    }

# 开始训练
def start_training(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, eval_dataset: Dataset, train_dataset: Dataset, max_length: int) -> None:
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
        bf16 = True,
        optim = OPTIMIZER,
        weight_decay = WEIGHT_DECAY,
        learning_rate = LEARNING_RATE,
        max_steps = MAX_STEPS,
        warmup_steps = int(WARMUP_STEPS),
        lr_scheduler_type = "cosine",
        per_device_eval_batch_size = EVAL_SIZE,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_checkpointing = GRADIENT_CHECKPOINTING,
        gradient_accumulation_steps = int(max(BATCH_SIZE, GRADIENT_ACCUMULATION_SIZE) / BATCH_SIZE),
        dataloader_pin_memory = True,
        dataloader_num_workers = min(8, os.cpu_count()),
        dataloader_persistent_workers = False,
    )

    trainer = Trainer(
        args = training_args,
        model = model,
        data_collator = DataCollatorForTokenClassification(
            tokenizer = tokenizer,
            padding = "max_length",
            max_length = max_length,
            pad_to_multiple_of = 8,
        ),
        eval_dataset = eval_dataset,
        train_dataset = train_dataset,
        compute_metrics = lambda eval_prediction: compute_metrics(eval_prediction = eval_prediction, id2label = model.config.id2label),
        processing_class = tokenizer,
    )
    trainer.add_callback(
        NERTrainerCallback(
            trainer = trainer,
            patience = PATIENCE,
        ),
    )
    trainer.add_callback(
        MemoryCallback(
            threshold = 0,
            check_steps = 0,
            force_clean_on_start = True,
        )
    )

    # 开始训练
    trainer.train()

# 主函数
def main() -> None:
    # 固定随机种子
    random.seed(SEED)

    # 加载分词器
    tokenizer = load_tokenizer()

    # 加载数据集
    eval_dataset, train_dataset, id2label, label2id, max_length = load_dataset(tokenizer)

    # 加载模型
    model = load_model(id2label, label2id)

    # 设置模型层
    set_layers(model)

    # 打印模型的参数量
    print_model_parameters(model)

    # 设置 wandb
    if WANDB_ENABLE == True:
        import wandb
        wandb.init(
            project = "NER",
            name = PROJECT_NAME,
        )

    # 开始任务
    start_training(model, tokenizer, eval_dataset, train_dataset, max_length)

    # 结束 wandb
    wandb.finish() if WANDB_ENABLE == True else None

# 主函数
if __name__ == "__main__":
    main()