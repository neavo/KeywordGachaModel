from rich import print
from transformers import Trainer
from transformers import TrainerState
from transformers import TrainerControl
from transformers import TrainerCallback
from transformers import TrainingArguments

class LRSearchCallback(TrainerCallback):

    def __init__(self, trainer: Trainer) -> None:
        super().__init__()

        # 初始化
        self.trainer = trainer

    # 步骤结束时
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        # 更新学习率调度器
        self.trainer.lr_scheduler.base_lrs = [param_group.get("lr") * 1.05 for param_group in self.trainer.optimizer.param_groups]
        self.trainer.lr_scheduler.step()