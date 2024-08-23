import os
import json
import shutil
import matplotlib.pyplot as plt  # 使用 matplotlib 绘图

import torch

from rich import box
from rich import print
from rich.table import Table
from rich.console import Console
from dataclasses import asdict
from transformers import TrainerCallback

class NERTrainerCallback(TrainerCallback):
    def __init__(self, model_name, patience, patience_keeper):
        self.console = Console()

        self.model_name = model_name
        self.patience = int(patience)               # 早停耐心值，即最大允许没有改进的轮数
        self.patience_keeper = int(patience_keeper) # 早停静默轮次，即前x轮不触发早停
        self.wait_for_early_stop = 0
        self.best_metric_for_save = -float("inf")
        self.best_metric_for_eval_loss = float("inf")
        self.best_metric_for_train_loss = float("inf")
        self.best_metric_for_f1 = -float("inf")

        # 初始化记录每种指标的历史数据
        self.training_epochs = []
        self.metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "f1": [],
            "recall": [],
            "precision": []
        }
        
    def set_trainer(self, trainer):
        self.trainer = trainer

    # 在训练开始时检查并移除旧的模型保存目录
    def on_train_begin(self, args, state, control, **kwargs):
        self.tokenizer = kwargs.get("tokenizer")
        self.best_path = f"{args.output_dir}/{self.model_name.replace("-", "_")}_ner_best"
        self.lastest_path = f"{args.output_dir}/{self.model_name.replace("-", "_")}_ner_latest"

        shutil.rmtree(self.best_path, ignore_errors = True)
        shutil.rmtree(self.lastest_path, ignore_errors = True)
        os.makedirs(self.best_path, exist_ok = True)
        os.makedirs(self.lastest_path, exist_ok = True)

    # 评估时
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 先更新指标
        self.update_metrics(args, state, control, metrics, **kwargs)

        # 再执行后续步骤
        self.save_lastest(args, state, control, metrics, **kwargs)
        self.check_and_save_best(args, state, control, metrics, **kwargs)
        self.check_early_stopping(args, state, control, metrics, **kwargs)

        return control

    # 结束训练时 
    def on_train_end(self, args, state, control, **kwargs):
        self.trainer.evaluate()

    # 逆序从字典或者自定义对象中查找第一个匹配的键对应的值
    def find_first_match_from_end(self, d, target_key, default = None):
        # 定义一个内部函数来递归地搜索值
        def search_value(value):
            # 如果值是字典，并且包含目标键，则返回该值
            if isinstance(value, dict) and target_key in value:
                return value[target_key]
            # 如果值是列表，则递归地在列表中查找
            elif isinstance(value, list):
                for item in reversed(value):
                    result = search_value(item)
                    if result is not None:
                        return result
            # 如果值是自定义对象，并且对象有属性或方法返回目标键的值
            elif hasattr(value, target_key):
                return getattr(value, target_key)
            # 如果属性值是字典或列表，递归地在其中查找
            elif isinstance(getattr(value, target_key, None), (dict, list)):
                return search_value(getattr(value, target_key))

        # 从嵌套结构的结尾开始查找
        if isinstance(d, (dict, list)):
            for item in reversed(d):
                result = search_value(item)
                if result is not None:
                    return result
        elif hasattr(d, target_key):
            return search_value(d)

        # 如果没有找到匹配项，返回默认值
        return default

    # 更新评估指标并保存
    def update_metrics(self, args, state, control, metrics, **kwargs):
        # 更新指标历史和训练轮数
        self.metrics_history["train_loss"].append(self.find_first_match_from_end(state.log_history, "loss", float("inf")))
        self.metrics_history["eval_loss"].append(metrics.get("eval_loss", float("inf")))
        self.metrics_history["f1"].append(metrics.get("eval_f1", 0))
        self.metrics_history["recall"].append(metrics.get("eval_recall", 0))
        self.metrics_history["precision"].append(metrics.get("eval_precision", 0))
        self.training_epochs.append(state.epoch)  # 记录训练轮数

        # 创建图形
        plt.figure(figsize = (12, 8))

        # 绘制所有指标
        plt.plot(self.training_epochs, self.metrics_history["train_loss"], label="Train Loss", color="blue")
        plt.plot(self.training_epochs, self.metrics_history["eval_loss"], label="Eval Loss", color="orange")
        plt.plot(self.training_epochs, self.metrics_history["f1"], label="F1 Score", color="green")
        plt.plot(self.training_epochs, self.metrics_history["recall"], label="Recall", color="red")
        plt.plot(self.training_epochs, self.metrics_history["precision"], label="Precision", color="purple")

        plt.title("Metrics Trends")
        plt.xlabel("Training Epochs")
        plt.ylabel("Value")
        plt.legend()  # 添加图例

        # 保存图表到文件
        plt.savefig(f"{self.best_path}/metrics_trends.png")
        plt.savefig(f"{self.lastest_path}/metrics_trends.png")
        plt.close()  # 关闭图形，以释放资源

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
    def save_lastest(self, args, state, control, metrics, **kwargs):
        # 保存最新模型到指定目录
        self.trainer.save_model(self.lastest_path)
        self.tokenizer.save_pretrained(self.lastest_path)

        # 保存评估信息
        metrics["train_loss"] = self.metrics_history["train_loss"][-1]
        metrics_file = os.path.join(self.lastest_path, "metrics.json")
        with open(metrics_file, "w", encoding = "utf-8") as file:
            json.dump(metrics, file, indent = 4, ensure_ascii = True)

        # 保存训练参数
        training_args_file = os.path.join(self.lastest_path, "training_args.json")
        with open(training_args_file, "w", encoding = "utf-8") as file:
            json.dump(asdict(args), file, indent = 4, ensure_ascii = True)

    # 判断是否需要保存最佳模型
    def check_and_save_best(self, args, state, control, metrics, **kwargs):
        key_metrics = self.metrics_history["f1"][-1]

        if key_metrics > self.best_metric_for_save:
            self.best_metric_for_save = key_metrics

            self.trainer.save_model(self.best_path)
            self.tokenizer.save_pretrained(self.best_path)

            # 保存评估信息
            metrics["train_loss"] = self.metrics_history["train_loss"][-1]
            metrics_file = os.path.join(self.best_path, "metrics.json")
            with open(metrics_file, "w", encoding = "utf-8") as file:
                json.dump(metrics, file, indent = 4, ensure_ascii = True)

            # 保存训练参数
            training_args_file = os.path.join(self.best_path, "training_args.json")
            with open(training_args_file, "w", encoding = "utf-8") as file:
                json.dump(asdict(args), file, indent = 4, ensure_ascii = True)

    # 判断是否需要触发早停
    def check_early_stopping(self, args, state, control, metrics, **kwargs):
        if control.should_training_stop:
            return

        key_metrics_f1 = self.metrics_history["f1"][-1]
        key_metrics_eval_loss = self.metrics_history["eval_loss"][-1]
        key_metrics_train_loss = self.metrics_history["train_loss"][-1]

        f1_improved = key_metrics_f1 > self.best_metric_for_f1
        eval_loss_improved = key_metrics_eval_loss < self.best_metric_for_eval_loss
        train_loss_improved = key_metrics_train_loss < self.best_metric_for_train_loss

        if f1_improved:
            print(""
                + f"在本次评估中，最佳评估指标已更新 "
                + f"{key_metrics_f1:.4f} / {self.best_metric_for_f1:.4f} ..."
            )
            self.wait_for_early_stop = 0
            self.best_metric_for_f1 = key_metrics_f1

        if eval_loss_improved:
            print(""
                + f"在本次评估中，最佳评估损失已更新 "
                + f"{key_metrics_eval_loss:.4f} / {self.best_metric_for_eval_loss:.4f} ..."
            )
            self.wait_for_early_stop = 0
            self.best_metric_for_eval_loss = key_metrics_eval_loss

        if train_loss_improved:
            print(""
                + f"在本次评估中，最佳训练损失已更新 "
                + f"{key_metrics_train_loss:.4f} / {self.best_metric_for_train_loss:.4f} ..."
            )
            self.best_metric_for_train_loss = key_metrics_train_loss

        if f1_improved or eval_loss_improved or train_loss_improved:
            print(f"")

        if not f1_improved and not eval_loss_improved:
            self.wait_for_early_stop += 1
            print(""
                + f"在本次评估中，"
                + f"评估指标为 {key_metrics_f1:.4f} / {self.best_metric_for_f1:.4f}，"
                + f"评估损失为 {key_metrics_eval_loss:.4f} / {self.best_metric_for_eval_loss:.4f}，"
                + f"训练损失为 {key_metrics_train_loss:.4f} / {self.best_metric_for_train_loss:.4f}，"
                + f"耐心计数器值为 {self.wait_for_early_stop:.4f} ..."
            )
            print(f"")

        # 如果等待时间超过耐心值，则触发早停
        if (
            state.epoch > self.patience_keeper
            and self.wait_for_early_stop >= self.patience
            and self.best_metric_for_eval_loss > self.best_metric_for_train_loss
        ):
            control.should_training_stop = True
            self.wait_for_early_stop = self.wait_for_early_stop - 1
            print(f"在连续 {self.patience} 次的评估中，各项指标均未改善，训练已中止 ...")
            print(f"")