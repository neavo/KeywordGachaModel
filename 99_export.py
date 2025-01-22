import argparse
import os
import sys
import shutil
import subprocess

import torch
from tqdm import tqdm
from rich import print

from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import BitsAndBytesConfig
from transformers import AutoModelForTokenClassification

# 加载模型
def load_model(target: str, output_path: str) -> PreTrainedModel:
    if "bf16" in output_path:
        return AutoModelForTokenClassification.from_pretrained(
            target,
            torch_dtype = torch.bfloat16,
            local_files_only = True,
            trust_remote_code = True,
            ignore_mismatched_sizes = True,
        )
    elif "bnb_4bit" in output_path:
        return AutoModelForTokenClassification.from_pretrained(
            target,
            local_files_only = True,
            trust_remote_code = True,
            ignore_mismatched_sizes = True,
            quantization_config = BitsAndBytesConfig(load_in_4bit = True),
        )
    elif "bnb_8bit" in output_path:
        return AutoModelForTokenClassification.from_pretrained(
            target,
            local_files_only = True,
            trust_remote_code = True,
            ignore_mismatched_sizes = True,
            quantization_config = BitsAndBytesConfig(load_in_8bit = True),
        )
    else:
        return AutoModelForTokenClassification.from_pretrained(
            target,
            local_files_only = True,
            trust_remote_code = True,
            ignore_mismatched_sizes = True,
            torch_dtype = torch.bfloat16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

def export_bnb_4bit(target: str) -> None:
    path = f"{target}_bnb_4bit"

    print(f"")
    print(f"正在导出 {path} ...")
    shutil.rmtree(f"{path}", ignore_errors = True)
    shutil.copytree(target, f"{path}", dirs_exist_ok = True)
    os.remove(f"{path}/model.safetensors") if os.path.exists(f"{path}/model.safetensors") else None
    os.remove(f"{path}/pytorch_model.bin") if os.path.exists(f"{path}/pytorch_model.bin") else None

    load_model(target, path).save_pretrained(f"{path}")

def export_bf16(target: str) -> None:
    path = f"{target}_bf16"

    print(f"")
    print(f"正在导出 {path} ...")
    shutil.rmtree(f"{path}", ignore_errors = True)
    shutil.copytree(target, f"{path}", dirs_exist_ok = True)
    os.remove(f"{path}/model.safetensors") if os.path.exists(f"{path}/model.safetensors") else None
    os.remove(f"{path}/pytorch_model.bin") if os.path.exists(f"{path}/pytorch_model.bin") else None

    load_model(target, path).save_pretrained(f"{path}")

def export_bnb_8bit(target: str) -> None:
    path = f"{target}_bnb_8bit"

    print(f"")
    print(f"正在导出 {path} ...")
    shutil.rmtree(f"{path}", ignore_errors = True)
    shutil.copytree(target, f"{path}", dirs_exist_ok = True)
    os.remove(f"{path}/model.safetensors") if os.path.exists(f"{path}/model.safetensors") else None
    os.remove(f"{path}/pytorch_model.bin") if os.path.exists(f"{path}/pytorch_model.bin") else None

    load_model(target, path).save_pretrained(f"{path}")

def export_onnx(target: str) -> None:
    path = f"{target}_onnx"

    print(f"")
    print(f"正在导出 {path} ...")
    shutil.rmtree(f"{path}", ignore_errors = True)
    shutil.copytree(target, f"{path}", dirs_exist_ok = True)
    os.remove(f"{path}/model.safetensors") if os.path.exists(f"{path}/model.safetensors") else None
    os.remove(f"{path}/pytorch_model.bin") if os.path.exists(f"{path}/pytorch_model.bin") else None

    subprocess.run(
        f"optimum-cli export onnx --task token-classification -m {target} {path}",
        shell = True,
        check = True,
    )

def export_onnx_avx512(target: str) -> None:
    path = f"{target}_onnx_avx512"

    print(f"")
    print(f"正在导出 {path} ...")
    shutil.rmtree(f"{path}", ignore_errors = True)
    shutil.copytree(target, f"{path}", dirs_exist_ok = True)
    os.remove(f"{path}/model.safetensors") if os.path.exists(f"{path}/model.safetensors") else None
    os.remove(f"{path}/pytorch_model.bin") if os.path.exists(f"{path}/pytorch_model.bin") else None

    subprocess.run(
        f"optimum-cli onnxruntime quantize --avx512 --per_channel --onnx_model {target}_onnx -o {path}",
        shell = True,
        check = True,
    )

# 运行主函数
def main(target: str) -> None:
    export_bf16(target)

    # export_bnb_4bit(target)
    # export_bnb_8bit(target)

    export_onnx(target)
    export_onnx_avx512(target)

# 运行主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type = str, help = "目标路径")
    args = parser.parse_args()

    main(args.target)