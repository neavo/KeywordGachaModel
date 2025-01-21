import json
import dataclasses
import os

import torch
from rich import print
from transformers import Trainer
from transformers import TrainerState
from transformers import TrainerControl
from transformers import TrainerCallback
from transformers import TrainingArguments

class PreTrainerCallback(TrainerCallback):

    def __init__(self, trainer: Trainer) -> None:
        super().__init__()

        # 初始化
        self.trainer = trainer
        self.tokenizer = trainer.processing_class

    # 训练开始时
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: dict) -> None:
        self.best_path = f"{args.output_dir}/best"
        self.lastest_path = f"{args.output_dir}/latest"

    # 结束训练时
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: dict) -> None:
        self.trainer.evaluate()

    # 评估时
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs: dict) -> None:
        # 保存最佳
        self.save_best(args, state, metrics, self.lastest_path)

        # 保存最新
        self.save_latest(args, state, metrics, self.lastest_path)

    # 保存到本地
    def save(self, args: TrainingArguments, state: TrainerState, metrics: dict, path: str) -> None:
        # 保存到指定目录
        self.trainer.save_model(path)
        self.trainer._save_optimizer_and_scheduler(path)
        self.tokenizer.save_pretrained(path)

        # 保存训练参数
        with open(f"{path}/training_args.json", "w", encoding = "utf-8") as writer:
            writer.write(json.dumps(args.to_dict(), indent = 4, ensure_ascii = True))

        # 保存训练状态
        with open(f"{path}/trainer_state.json", "w", encoding = "utf-8") as writer:
            writer.write(json.dumps(dataclasses.asdict(state), indent = 4, ensure_ascii = True))

    # 保存最佳
    def save_best(self, args: TrainingArguments, state: TrainerState, metrics: dict, path: str) -> None:
        eval_loss_now = self.get_best_eval_loss_from_state(dataclasses.asdict(state))

        try:
            with open(f"{self.best_path}/trainer_state.json", "r", encoding = "utf-8") as reader:
                eval_loss_best = self.get_best_eval_loss_from_state(json.load(reader))
        except Exception as e:
            eval_loss_best = float("inf")

        if eval_loss_now < eval_loss_best:
            print(""
                + f"\n\n\n"
                + f"在本次评估中，最佳评估损失已更新 {eval_loss_best:.4f} -> {eval_loss_now:.4f} ..."
                + f"\n"
            )
            self.save(args, state, metrics, self.best_path)

    # 保存最新
    def save_latest(self, args: TrainingArguments, state: TrainerState, metrics: dict, path: str) -> None:
        self.save(args, state, metrics, self.lastest_path)

    # 获取最佳评估损失
    def get_best_eval_loss_from_state(self, state: dict) -> float:
        result = [
            item.get("eval_loss")
            for item in state.get("log_history", [])
            if "eval_loss" in item and isinstance(item.get("eval_loss"), (int, float))
        ]

        return min(result, default = float("inf"))