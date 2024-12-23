import re
import json
import random
from datetime import datetime

from rich import print

import numpy
import wandb
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

from model.NERTrainerCallback import NERTrainerCallback

# 模型
MODEL_NAME = "modern_bert"
MODEL_PATH = f"assets/{MODEL_NAME}"
OUTPUT_PATH = "output"

# 训练
EPOCHS = 100
PATIENCE = 100
PATIENCE_KEEPER = 0
EVAL_SIZE = 256
BATCH_SIZE = 32
GRADIENT_CHECKPOINTING = False
GRADIENT_ACCUMULATION_SIZE = 0
FROZEN_LAYER = 0
LEARNING_RATE = 5 * 1e-5

# 输出
LOG_STEPS = 5
INTERVAL_STEPS = 100

# 数据
DO_LOWER_CASE = False
DATASET_PATH = [
    ("dataset/ner/zh_1.json", 99 * 10000),
    ("dataset/ner/en_1.json", 99 * 10000),
    ("dataset/ner/jp_1.json", 99 * 10000),
    ("dataset/ner/ko_1.json", 99 * 10000),
]

# 加载模型
def load_model(id2label: dict, label2id: dict) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.id2label = id2label
    config.label2id = label2id
    config.num_labels = len(id2label)

    if "modern" in MODEL_NAME:
        return AutoModelForTokenClassification.from_pretrained(
            MODEL_PATH,
            config = config,
            local_files_only = True,
            trust_remote_code = True,
            ignore_mismatched_sizes = True,
            torch_dtype = torch.bfloat16,
            attn_implementation = "flash_attention_2",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return AutoModelForTokenClassification.from_pretrained(
            MODEL_PATH,
            config = config,
            local_files_only = True,
            trust_remote_code = True,
            ignore_mismatched_sizes = True,
            torch_dtype = torch.bfloat16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

# 加载分词器
def load_tokenizer() -> PreTrainedTokenizerFast:
    if any(v in MODEL_NAME for v in ("bloom", "gpt2", "roberta", "deberta")):
        return AutoTokenizer.from_pretrained(
            MODEL_PATH,
            do_lower_case = DO_LOWER_CASE,
            add_prefix_space = True,
            local_files_only = True,
        )
    else:
        return AutoTokenizer.from_pretrained(
            MODEL_PATH,
            do_lower_case = DO_LOWER_CASE,
            local_files_only = True,
        )

# 加载数据集
def load_dataset(tokenizer: PreTrainedTokenizerFast) -> tuple[Dataset, Dataset, dict, dict]:
    data = []
    for path, num in DATASET_PATH:
        with open(path, "r", encoding = "utf-8") as file:
            data_ex = json.load(file)
            data.extend(random.sample(data_ex, min(int(num), len(data_ex))))

    # 只取需要的字段，避免后续转换格式时的错误
    data = [{"sentence": v.get("sentence", ""), "entities": v.get("entities", [])} for v in data]

    # 生成 id-label 映射表
    types = set()
    for v in data:
        for entity in v.get("entities", []):
            if entity["ner_type"] != "":
                types.add(entity["ner_type"])
    id2label = {0: "O"}
    for c in list(sorted(types)):
        id2label[len(id2label)] = f"B-{c}"
        id2label[len(id2label)] = f"I-{c}"
    label2id = {v: k for k, v in id2label.items()}

    # 生成数据集
    dataset_tokenized = Dataset.from_list(data).map(
        lambda samples: load_dataset_map_function(samples, tokenizer, label2id),
        num_proc = 1,
        batched = True,
        batch_size = 1024,
        remove_columns = ["sentence", "entities"],
    )

    # 拆分数据集
    dataset_dict = dataset_tokenized.train_test_split(
        seed = 42,
        shuffle = True,
        test_size = 2048,
        keep_in_memory = True,
        load_from_cache_file = False,
    )

    eval_dataset, train_dataset = dataset_dict.get("test"), dataset_dict.get("train")

    print("")
    print("数据加载已完成 ... 样本如下：")
    print("")
    print_dataset_sample(dataset_tokenized, id2label)
    print("")
    print(f"加载数据文件 {len(DATASET_PATH)} 个，共 {len(data)} 条数据 ...")
    print(f"测试集 {len(eval_dataset)} 条数据，其中最大长度为 {max(len(v.get("input_ids")) for v in eval_dataset)} ...")
    print(f"训练集 {len(train_dataset)} 条数据，其中最大长度为 {max(len(v.get("input_ids")) for v in train_dataset)} ...")

    return eval_dataset, train_dataset, id2label, label2id

# 打印数据集样本
def print_dataset_sample(dateset: Dataset, id2label: dict) -> None:
    if len(dateset) == 0:
        return

    labels = dateset[0].get("labels")
    input_ids = dateset[0].get("input_ids")
    input_tokens = dateset[0].get("input_tokens")

    print(f"{"input_tokens":<8}\t\t{"labels":<4}\t\t{"input_ids":<4}")
    for x, y, z in zip(input_tokens, labels, input_ids):
        print(f"{x:<8}\t\t{id2label.get(y):<4}\t\t{z:<4}")

# 数据集映射函数
def load_dataset_map_function(samples: dict, tokenizer: PreTrainedTokenizerFast, label2id: dict) -> BatchEncoding:
    encodings = tokenizer(
        samples.get("sentence"),
        return_attention_mask = True,
        return_offsets_mapping = True,
        return_special_tokens_mask = True,
    )

    # 生成 input_tokens
    encodings["input_tokens"] = []
    for tokens in [tokenizer.convert_ids_to_tokens(v) for v in encodings.input_ids]:
        encodings["input_tokens"].append([])
        for t in tokens:
            encodings["input_tokens"][-1].append(tokenizer.convert_tokens_to_string([t]))

    # 生成 labels
    for i, _ in enumerate(encodings.get("input_ids")):
        sentence = samples.get("sentence", [])[i]
        entities = samples.get("entities", [])[i]
        input_ids = encodings.get("input_ids")[i]
        offsets_mapping = encodings.get("offset_mapping")[i]

        # 遍历实体词语
        result = []
        for entity in entities:
            name = entity.get("name", "")
            ner_type = entity.get("ner_type", "")

            # 获取实体词语在字符串中的位置
            char_start = sentence.find(name)
            char_end = char_start + len(name)

            # 检查实体是否在字符串中
            # if char_start < 0:
            #     print(
            #         (
            #             "\n"
            #             + f"{name}" + "\n"
            #             + "[green]-->[/]"
            #             + f"{sentence}" + "\n"
            #             + "\n"
            #         )
            #     )

            # 有效性检查
            if char_start < 0 or name == "" or ner_type == "":
                continue

            # 通过字符位置反查 Token 位置
            token_start, token_end = char_offset_to_token_offset(char_start, char_end, offsets_mapping)

            # 跳过不存在的 Token
            if token_start == -1 or token_end == -1:
                continue

            result.append((token_start, token_end, ner_type))

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
    print(f"{MODEL_NAME} : layer - {layer / 1e6:.2f} M / embedding - {embedding / 1e6:.2f} M / total - {total / 1e6:.2f} M")
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
        "f1": f1_score(true_labels, true_predictions, mode = "strict", average = "weighted", zero_division = 0),
        "recall": recall_score(true_labels, true_predictions, mode = "strict", average = "weighted", zero_division = 0),
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions, mode = "strict", average = "weighted", zero_division = 0),
        "classification_report": classification_report(true_labels, true_predictions, mode = "strict", output_dict = True, zero_division = 0)
    }

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
        save_strategy = "no",

        # 训练
        bf16 = True,
        bf16_full_eval = True,
        # optim = "ademamix_8bit",
        optim = "ademamix",
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

    callback = NERTrainerCallback(
        model_name = MODEL_NAME,
        patience = PATIENCE,
        patience_keeper = PATIENCE_KEEPER,
    )

    trainer = Trainer(
        args = training_args,
        model = model,
        callbacks = [callback],
        data_collator = DataCollatorForTokenClassification(
            tokenizer = tokenizer,
            padding = "longest",
            pad_to_multiple_of = 8,
        ),
        eval_dataset = eval_dataset,
        train_dataset = train_dataset,
        compute_metrics = lambda eval_prediction: compute_metrics(eval_prediction = eval_prediction, id2label = model.config.id2label),
        processing_class = tokenizer,
    )

    # 设置回调中的 trainer 属性
    callback.set_trainer(trainer)

    # 开始训练
    trainer.train()

# 主函数
def main() -> None:
    # 固定随机种子
    random.seed(42)

    # 加载分词器
    tokenizer = load_tokenizer()

    # 加载数据集
    eval_dataset, train_dataset, id2label, label2id = load_dataset(tokenizer)

    # 加载模型
    model = load_model(id2label, label2id)

    # 设置模型层
    set_layers(model)

    # 打印模型的参数量
    print_model_parameters(model)

    # 设置 wandb
    wandb.init(
        project = "NER",
        name = f"{MODEL_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}",
    )

    # 开始训练
    start_training(model, tokenizer, eval_dataset, train_dataset)

# 主函数
if __name__ == "__main__":
    main()