import os
import shutil
import argparse

import torch
from rich import print

from transformers import AutoConfig
from transformers import PreTrainedModel
from transformers import AutoModelForTokenClassification

# 加载模型
def load_model(target: str, output_path: str) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(
        target,
        local_files_only = True,
        trust_remote_code = True,
    )
    config.reference_compile = None

    if "bf16" in output_path:
        return AutoModelForTokenClassification.from_pretrained(
            target,
            config = config,
            torch_dtype = torch.bfloat16,
            attn_implementation = "sdpa",
            local_files_only = True,
            trust_remote_code = True,
            ignore_mismatched_sizes = True,
        )
    elif "fp16" in output_path:
        return AutoModelForTokenClassification.from_pretrained(
            target,
            config = config,
            torch_dtype = torch.float16,
            attn_implementation = "sdpa",
            local_files_only = True,
            trust_remote_code = True,
            ignore_mismatched_sizes = True,
        )
    else:
        return AutoModelForTokenClassification.from_pretrained(
            target,
            config = config,
            torch_dtype = torch.float32,
            attn_implementation = "sdpa",
            local_files_only = True,
            trust_remote_code = True,
            ignore_mismatched_sizes = True,
        )

# 导出模型
def export(input_path: str, dtype: str) -> None:
    output_path = f"{input_path}_{dtype}"

    print("")
    print(f"正在导出 [green]{output_path}[/] ...")
    shutil.rmtree(f"{output_path}", ignore_errors = True)
    shutil.copytree(input_path, f"{output_path}", dirs_exist_ok = True)
    os.remove(f"{output_path}/model.safetensors") if os.path.exists(f"{output_path}/model.safetensors") else None
    os.remove(f"{output_path}/pytorch_model.bin") if os.path.exists(f"{output_path}/pytorch_model.bin") else None

    load_model(input_path, output_path).save_pretrained(f"{output_path}")

# 运行主函数
def main(target: str) -> None:
    export(target, "bf16")

# 运行主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type = str, help = "目标路径")
    args = parser.parse_args()

    main(args.target)