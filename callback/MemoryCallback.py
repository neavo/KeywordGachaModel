import os

import torch
from transformers import TrainerState
from transformers import TrainerControl
from transformers import TrainerCallback
from transformers import TrainingArguments

class MemoryCallback(TrainerCallback):

    def __init__(self, threshold: float, check_steps: int, force_clean_on_start: bool) -> None:
        super().__init__()

        # 初始化
        self.threshold = threshold
        self.check_steps = check_steps
        self.force_clean_on_start = force_clean_on_start

    # 步骤结束时
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if state.global_step == 8 and self.force_clean_on_start == True:
            self.clear_memory(0.00)
        elif self.check_steps > 0 and state.global_step % self.check_steps == 0:
            self.clear_memory(self.threshold)

    # 清理显存
    def clear_memory(self, threshold: float) -> None:
        # 使用 nvidia-smi 获取显存信息
        result = os.popen("nvidia-smi --query-gpu=memory.total,memory.reserved,memory.used --format csv,noheader,nounits").readlines()
        result = result[0].strip().split(", ")
        total = int(result[0])
        used = int(result[1]) + int(result[2])

        # 如果显存使用量大于阈值，则清理显存
        if used / total > threshold:
            torch.cuda.empty_cache()