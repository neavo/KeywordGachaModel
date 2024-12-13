import os
import json
import shutil

from rich import box
from rich import print
from rich.table import Table
from rich.console import Console
from dataclasses import asdict
from transformers import Trainer
from transformers import TrainerState
from transformers import TrainerControl
from transformers import TrainerCallback
from transformers import TrainingArguments

class NERTrainerCallback(TrainerCallback):

    def __init__(self, model_name: str, patience: int, patience_keeper: int) -> None:

        # 初始化
        self.model_name = model_name                    # 模型名称
        self.patience = int(patience)                   # 早停耐心值，即最大允许没有改进的轮数
        self.patience_keeper = int(patience_keeper)     # 早停静默轮次，即前x轮不触发早停

        self.console = Console()
        self.wait_for_early_stop = 0
        self.best_metric_for_save = -float("inf")
        self.best_metric_for_eval_loss = float("inf")
        self.best_metric_for_train_loss = float("inf")
        self.best_metric_for_f1 = -float("inf")

        # 初始化记录每种指标的历史数据
        self.metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "f1": [],
            "recall": [],
            "precision": []
        }

    def set_trainer(self, trainer: Trainer) -> None:
        self.trainer = trainer
        self.tokenizer = trainer.processing_class

    # 在训练开始时检查并移除旧的模型保存目录
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: dict) -> None:
        self.best_path = f"{args.output_dir}/{self.model_name.replace("-", "_")}_ner_best"
        self.lastest_path = f"{args.output_dir}/{self.model_name.replace("-", "_")}_ner_latest"

        shutil.rmtree(self.best_path, ignore_errors = True)
        shutil.rmtree(self.lastest_path, ignore_errors = True)
        os.makedirs(self.best_path, exist_ok = True)
        os.makedirs(self.lastest_path, exist_ok = True)

    # 评估时
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs: dict) -> None:
        # 先更新指标
        self.metrics_history["train_loss"] = self.generate_train_loss_metrics(args, state, control, metrics, **kwargs)
        self.metrics_history["eval_loss"].append(metrics.get("eval_loss", float("inf")))
        self.metrics_history["f1"].append(metrics.get("eval_f1", 0))
        self.metrics_history["recall"].append(metrics.get("eval_recall", 0))
        self.metrics_history["precision"].append(metrics.get("eval_precision", 0))

        # 再执行后续步骤
        self.print_log(args, state, control, metrics, **kwargs)
        self.save_lastest(args, state, control, metrics, **kwargs)
        self.check_and_save_best(args, state, control, metrics, **kwargs)
        self.check_early_stopping(args, state, control, metrics, **kwargs)

    # 结束训练时
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: dict) -> None:
        self.trainer.evaluate()

    # 打印评估信息
    def print_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs: dict) -> None:
        # 打印表格到控制台
        table = Table(box = box.ASCII2, expand = True, highlight = True, show_lines = True, show_header = False, border_style = "light_goldenrod2")
        table.add_column(justify = "left")
        table.add_column(justify = "right")
        table.add_column(justify = "left")
        table.add_column(justify = "right")
        table.add_column(justify = "left")
        table.add_column(justify = "right")

        # 将指标数据添加到表格中
        table.add_row(
            f"epoch",
            f"{float(metrics["epoch"]):>8.4f}",
            f"eval_loss",
            f"{float(metrics["eval_loss"]):>8.4f}",
            f"train_loss",
            f"{float(self.metrics_history["train_loss"][-1] if len(self.metrics_history["train_loss"]) > 0 else 1):>8.4f}",
        )
        table.add_row(
            f"eval_f1",
            f"{float(metrics["eval_f1"]):>8.4f}",
            f"eval_recall",
            f"{float(metrics["eval_recall"]):>8.4f}",
            f"eval_precision",
            f"{float(metrics["eval_precision"]):>8.4f}",
        )

        # 打印表格到控制台
        self.console.print("\n\n")
        self.console.print(table)
        self.console.print("")

    # 保存当前 模型、tokenizer 和 评估信息 到本地
    def save_lastest(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs: dict) -> None:
        # 保存最新模型到指定目录
        self.trainer.save_model(self.lastest_path)
        self.tokenizer.save_pretrained(self.lastest_path)

        # 保存评估信息
        metrics["train_loss"] = self.metrics_history["train_loss"][-1]
        metrics_file = os.path.join(self.lastest_path, "metrics.json")
        with open(metrics_file, "w", encoding = "utf-8") as writer:
            writer.write(json.dumps(metrics, indent = 4, ensure_ascii = True))

        # 保存训练参数
        training_args_file = os.path.join(self.lastest_path, "training_args.json")
        with open(training_args_file, "w", encoding = "utf-8") as writer:
            writer.write(json.dumps(asdict(args), indent = 4, ensure_ascii = True))

    # 判断是否需要保存最佳模型
    def check_and_save_best(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs: dict) -> None:
        key_metrics = self.metrics_history["f1"][-1]

        if key_metrics > self.best_metric_for_save:
            self.best_metric_for_save = key_metrics

            self.trainer.save_model(self.best_path)
            self.tokenizer.save_pretrained(self.best_path)

            # 保存评估信息
            metrics["train_loss"] = self.metrics_history["train_loss"][-1]
            metrics_file = os.path.join(self.best_path, "metrics.json")
            with open(metrics_file, "w", encoding = "utf-8") as writer:
                writer.write(json.dumps(metrics, indent = 4, ensure_ascii = True))

            # 保存训练参数
            training_args_file = os.path.join(self.best_path, "training_args.json")
            with open(training_args_file, "w", encoding = "utf-8") as writer:
                writer.write(json.dumps(asdict(args), indent = 4, ensure_ascii = True))

    # 判断是否需要触发早停
    def check_early_stopping(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs: dict) -> None:
        # 获取当前评估指标
        metrics_f1 = self.metrics_history["f1"][-1]
        metrics_eval_loss = self.metrics_history["eval_loss"][-1]
        metrics_train_loss = self.metrics_history["train_loss"][-1]

        # 判断评估指标情况
        f1_improved = metrics_f1 > self.best_metric_for_f1
        eval_loss_improved = metrics_eval_loss < self.best_metric_for_eval_loss
        train_loss_improved = metrics_train_loss < self.best_metric_for_train_loss

        # 更新最佳评估指标
        if f1_improved == True:
            print(f"在本次评估中，最佳评估指标已更新 {metrics_f1:.4f} / {self.best_metric_for_f1:.4f} ...")
            self.best_metric_for_f1 = metrics_f1

        # 更新最佳评估损失
        if eval_loss_improved == True:
            print(f"在本次评估中，最佳评估损失已更新 {metrics_eval_loss:.4f} / {self.best_metric_for_eval_loss:.4f} ...")
            self.best_metric_for_eval_loss = metrics_eval_loss

        # 更新最佳训练损失
        if train_loss_improved == True:
            print(f"在本次评估中，最佳训练损失已更新 {metrics_train_loss:.4f} / {self.best_metric_for_train_loss:.4f} ...")
            self.best_metric_for_train_loss = metrics_train_loss

        # 打印分隔行
        print("") if f1_improved == True or eval_loss_improved == True or train_loss_improved == True else None

        # 如果评估指标或评估损失有更新，则耐心计数值重置，否则耐心计数值增加
        if f1_improved == True or eval_loss_improved == True:
            self.wait_for_early_stop = 0
        else:
            self.wait_for_early_stop = self.wait_for_early_stop + 1
            print(
                "\n"
                + "在本次评估中，"
                + f"评估指标为 {metrics_f1:.4f} / {self.best_metric_for_f1:.4f}，"
                + f"评估损失为 {metrics_eval_loss:.4f} / {self.best_metric_for_eval_loss:.4f}，"
                + f"训练损失为 {metrics_train_loss:.4f} / {self.best_metric_for_train_loss:.4f}，"
                + f"耐心计数值为 {float(self.wait_for_early_stop):.4f} ..."
                + "\n"
            )

        # 如果轮数小于等于耐心保持值，则不触发早停
        if state.epoch <= self.patience_keeper:
            return

        # 如果耐心计数值小于耐心值，则不触发早停
        if self.wait_for_early_stop <= self.patience:
            return

        # 如果评估损失尚未与训练损失交叉，则不触发早停
        if self.best_metric_for_eval_loss <= self.best_metric_for_train_loss:
            return

        print(f"在连续 {self.patience} 次的评估中，目标指标均未改善，训练已中止 ...")
        print("")

        # 触发早停
        control.should_training_stop = True

    # 生成训练损失评估指标
    def generate_train_loss_metrics(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs: dict) -> None:
        return [v.get("loss") for v in state.log_history if "loss" in v]