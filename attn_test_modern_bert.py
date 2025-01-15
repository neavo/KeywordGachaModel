import os
import gc
import time
import json
import random

import torch
from rich import print
from datasets import Dataset
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import TrainerState
from transformers import TrainerControl
from transformers import TrainerCallback
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers.utils import is_torch_bf16_gpu_available
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

BATCH_SIZE = 8
MODEL_PATH = "assets/modern_bert_cjk"
ATTN = (
    # "sdpa",
    "flash_attention_2",
)

class Callback(TrainerCallback):

    def __init__(self) -> None:
        super().__init__()

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        self.start_time = time.time()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        self.used_time = time.time() - self.start_time

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        # 延迟计算显存，以避免显存波动等误差
        if state.global_step == 25:
            result = os.popen("nvidia-smi --query-gpu=memory.total,memory.reserved,memory.used --format csv,noheader,nounits").readlines()
            result = result[0].strip().split(", ")
            self.used_vram = int(result[1]) + int(result[2])

# 加载模型
def load_model(attn: str, torch_bf16: bool) -> PreTrainedModel:
    if torch_bf16 == False:
        return AutoModelForMaskedLM.from_config(
            AutoConfig.from_pretrained(MODEL_PATH),
            attn_implementation = attn,
            trust_remote_code = True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return AutoModelForMaskedLM.from_config(
            AutoConfig.from_pretrained(MODEL_PATH),
            attn_implementation = attn,
            torch_dtype = torch.bfloat16 if is_torch_bf16_gpu_available() == True else torch.float16,
            trust_remote_code = True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

# 加载分词器
def load_tokenizer() -> PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(
        MODEL_PATH,
        do_lower_case = False,
        local_files_only = True,
    )

# 加载数据集
def load_dataset(tokenizer: PreTrainedTokenizerFast) -> tuple[Dataset, Dataset]:
    print("")
    print("正在加载数据集 ...")
    print("")

    # 加载或者生成数据集
    dataset_tokenized = Dataset.from_text("dataset/pt/modern_bert_cjk_jp_r18_rpg.txt").map(
        lambda samples: load_dataset_map_function(samples, tokenizer),
        batched = True,
        remove_columns = ["text"],
        keep_in_memory = True,
    )

    print("")
    print(f"数据加载已完成，共加载 {len(dataset_tokenized)} 条数据 ...")
    print("")

    return dataset_tokenized

# 映射函数
def load_dataset_map_function(samples: dict, tokenizer: PreTrainedTokenizerFast) -> BatchEncoding:
    return tokenizer(
        samples.get("text"),
        padding = "max_length",
        truncation = True,
        max_length = 256,
        return_attention_mask = True,
        return_special_tokens_mask = True,
    )

# 开始训练
def start_training(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, train_dataset: Dataset, ga: int, optim: str) -> None:
    training_args = TrainingArguments(
        # 输出
        output_dir = "output",
        report_to = "none",
        eval_strategy = "no",
        save_strategy = "no",
        logging_strategy = "no",

        # 训练
        bf16 = True,
        optim = "paged_ademamix_32bit",
        num_train_epochs = 1,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_checkpointing = True,
        gradient_accumulation_steps = ga,
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
        train_dataset = train_dataset.select(range(0, BATCH_SIZE * 300)),
        processing_class = tokenizer,
    )

    callback = Callback()
    trainer.add_callback(callback)
    trainer.train()

    return callback.used_vram, callback.used_time

# 主函数
def main() -> None:
    # 固定随机种子
    random.seed(42)

    # 加载分词器
    tokenizer = load_tokenizer()

    # 加载数据集
    dataset_tokenized = load_dataset(tokenizer)

    # 开始测试
    logs = []
    for gradient_accumulation in (2, ):
            for optim in ("adamw_torch", "paged_adamw_32bit", "paged_adamw_8bit"):
                    for attn in ATTN:
                        # 运行多次以获取最佳值
                        best_used_vram = 65535
                        best_used_time = 65535
                        for _ in range(2):
                            # ensure previous run is complete
                            torch.cuda.synchronize()

                            # 清理内存
                            torch.cuda.empty_cache()
                            gc.collect()

                            # 加载模型
                            model = load_model(attn, True)

                            # 开始训练
                            used_vram, used_time = start_training(model, tokenizer, dataset_tokenized, gradient_accumulation, optim)

                            # 更新结果
                            if used_time < best_used_time:
                                best_used_vram = used_vram
                                best_used_time = used_time

                        # 添加结果
                        logs.append({
                            "gradient_accumulation": gradient_accumulation,
                            "attn": attn,
                            "optim": optim,
                            "used_vram": best_used_vram,
                            "used_time": best_used_time,
                        })


    # 打印结果
    from rich.pretty import pprint
    pprint(logs)
    with open("logs.json", "w", encoding = "utf-8") as writer:
        json.dump(logs, writer, indent = 4, ensure_ascii = False)

# 主函数
if __name__ == "__main__":
    main()