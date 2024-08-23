import os
import re
import json
import random
import functools
from datetime import datetime

from tqdm import tqdm
from rich import print

import numpy
import wandb
import evaluate
import bitsandbytes
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

import torch
from torch.utils.data import DataLoader

from seqeval.metrics import f1_score
from seqeval.metrics import recall_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import precision_score
from seqeval.metrics import classification_report
from sklearn.model_selection import train_test_split

from model.NERDataset import NERDataset
from model.NERTrainerCallback import NERTrainerCallback

# 参数设置
MODEL_NAME = "facebookai_xlm_roberta_base_pretrain_20240823"
MODEL_PATH = f"assets/{MODEL_NAME}"
OUTPUT_PATH = "output"
DATASET_PATH = "dataset/ner"
EPOCHS = 24
PATIENCE = 12
PATIENCE_KEEPER = 0
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_SIZE = 64
FROZEN_LAYER = 0
LEARNING_RATE = 2 * 1e-5
DO_LOWER_CASE = False
INTERVAL_STEPS = 200

# 工作模式
MODE_MEASUREMENT = False

# 加载分词器
def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_PATH,
        do_lower_case = DO_LOWER_CASE,
        local_files_only = True,
    )

# 加载数据集
def load_dataset(tokenizer):
    count = 0
    datas = []
    for file in os.scandir(DATASET_PATH):
        if file.name.endswith(".json"):
            with open(file.path, "r", encoding = "utf-8") as file:
                count = count + 1
                datas.extend(json.load(file))

    print(f"")
    print(f"找到数据文件 {count} 个，共 {len(datas)} 条数据 ...")

    # 分割数据集
    train_datas, test_datas = train_test_split(datas, test_size = 0.02, shuffle = True, random_state = 42)

    # 创建数据集和数据加载器
    print(f"")
    test_dataset = NERDataset(test_datas, tokenizer, MODE_MEASUREMENT)
    train_dataset = NERDataset(train_datas, tokenizer, MODE_MEASUREMENT)
    print(f"")
    print(f"[green]test_dataset[/] 中最长条目为 {test_dataset.max_lenght} ...")
    print(f"[green]train_dataloader[/] 中最长条目为 {train_dataset.max_lenght} ...")
    print(f"")

    # for sample in train_dataloader:
    #     for k, v in sample.items():
    #         print(f"{k} : {v}")
    #     raise
    # raise

    return test_dataset, train_dataset

# 加载模型
def load_model(test_dataset, train_dataset):
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.id2label = train_dataset.id2label
    config.label2id = train_dataset.label2id
    config.num_labels = len(train_dataset.id2label)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATH,
        config = config,
        local_files_only = True,
        ignore_mismatched_sizes = True
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    return model

# 设置模型层
def set_layers(model):
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
        project = f"NER",
        name = f"{MODEL_NAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}",
    )

# 计算评估指标
def compute_metrics(eval_prediction, test_dataset, train_dataset):
    predictions, labels = eval_prediction    
    predictions = numpy.argmax(predictions, axis = 2) # 对于 3 维张量， axis = 2 与 axis = -1 是一样的

    true_labels = [
        [train_dataset.id2label[l] for p, l in zip(pred, lab) if p != -100 and l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_predictions = [
        [train_dataset.id2label[p] for p, l in zip(pred, lab) if p != -100 and l != -100]
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
def start_training(model, tokenizer, test_dataset, train_dataset):
    training_args = TrainingArguments(
        optim = "adamw_8bit",
        output_dir = OUTPUT_PATH,
        warmup_ratio = 0.1,
        weight_decay = 0.01,
        learning_rate = LEARNING_RATE,
        logging_dir = "logs",   
        logging_steps = INTERVAL_STEPS / 10,     
        eval_steps = INTERVAL_STEPS,
        eval_strategy = "steps",
        save_strategy = "no",
        save_safetensors = False,
        num_train_epochs = EPOCHS,
        bf16 = True,
        bf16_full_eval = True,
        per_device_eval_batch_size = min(128, BATCH_SIZE * 4),
        per_device_train_batch_size = BATCH_SIZE,
        gradient_checkpointing = False,
        gradient_accumulation_steps = max(1, int(GRADIENT_ACCUMULATION_SIZE / BATCH_SIZE)),
    )

    trainer = Trainer(
        args = training_args,
        model = model,
        callbacks = [NERTrainerCallback(
            model_name = MODEL_NAME,
            patience = PATIENCE,
            patience_keeper = PATIENCE_KEEPER,
        )],
        tokenizer = tokenizer,
        optimizers = (
            bitsandbytes.optim.Adam8bit(
                model.parameters(),
                lr = LEARNING_RATE,
                weight_decay = 0.01,
            ),
            None,
        ),
        eval_dataset = test_dataset,
        train_dataset = train_dataset,
        compute_metrics = functools.partial(compute_metrics, test_dataset = test_dataset, train_dataset = train_dataset),
    )

    # 更新回调中的 trainer 属性
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, NERTrainerCallback):
            callback.set_trainer(trainer)  # 设置实际的 Trainer 实例

    trainer.train()

# 主函数
def main():
    # 加载分词器
    tokenizer = load_tokenizer()

    # 加载数据集
    test_dataset, train_dataset = load_dataset(tokenizer)

    # 测量模式时不继续后续流程
    if MODE_MEASUREMENT:
        return

    # 加载模型
    model = load_model(test_dataset, train_dataset)

    # 设置模型层
    set_layers(model)

    # 打印模型的参数量
    print_model_parameters(model)

    # 设置 wandb
    set_wandb()

    # 开始训练
    start_training(model, tokenizer, test_dataset, train_dataset)

# 主函数
if __name__ == "__main__":
    main()