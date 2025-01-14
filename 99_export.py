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
def load_model(input: str, output_path: str) -> PreTrainedModel:
    if "bnb_4bit" in output_path:
        return AutoModelForTokenClassification.from_pretrained(
            input,
            local_files_only = True,
            trust_remote_code = True,
            low_cpu_mem_usage = True,
            ignore_mismatched_sizes = True,
            torch_dtype = torch.bfloat16,
            quantization_config = BitsAndBytesConfig(load_in_4bit = True),
        )
    elif "bnb_8bit" in output_path:
        return AutoModelForTokenClassification.from_pretrained(
            input,
            local_files_only = True,
            trust_remote_code = True,
            low_cpu_mem_usage = True,
            ignore_mismatched_sizes = True,
            torch_dtype = torch.bfloat16,
            quantization_config = BitsAndBytesConfig(load_in_8bit = True),
        )
    else:
        return AutoModelForTokenClassification.from_pretrained(
            input,
            local_files_only = True,
            trust_remote_code = True,
            ignore_mismatched_sizes = True,
            torch_dtype = torch.bfloat16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

def export_bnb_4bit(tag) -> None:
    path = f"{tag}_bnb_4bit"

    print(f"")
    print(f"正在导出 {path} ...")
    shutil.rmtree(f"{path}", ignore_errors = True)
    shutil.copytree(tag, f"{path}", dirs_exist_ok = True)
    os.remove(f"{path}/model.safetensors") if os.path.exists(f"{path}/model.safetensors") else None
    os.remove(f"{path}/pytorch_model.bin") if os.path.exists(f"{path}/pytorch_model.bin") else None

    load_model(tag, path).save_pretrained(f"{path}")

def export_bnb_8bit(tag) -> None:
    path = f"{tag}_bnb_8bit"

    print(f"")
    print(f"正在导出 {path} ...")
    shutil.rmtree(f"{path}", ignore_errors = True)
    shutil.copytree(tag, f"{path}", dirs_exist_ok = True)
    os.remove(f"{path}/model.safetensors") if os.path.exists(f"{path}/model.safetensors") else None
    os.remove(f"{path}/pytorch_model.bin") if os.path.exists(f"{path}/pytorch_model.bin") else None

    load_model(tag, path).save_pretrained(f"{path}")

def export_onnx(tag: str) -> None:
    path = f"{tag}_onnx"

    print(f"")
    print(f"正在导出 {path} ...")
    shutil.rmtree(f"{path}", ignore_errors = True)
    shutil.copytree(tag, f"{path}", dirs_exist_ok = True)
    os.remove(f"{path}/model.safetensors") if os.path.exists(f"{path}/model.safetensors") else None
    os.remove(f"{path}/pytorch_model.bin") if os.path.exists(f"{path}/pytorch_model.bin") else None

    subprocess.run(
        f"optimum-cli export onnx --task token-classification -m {tag} {path}",
        shell = True,
        check = True,
    )

def export_onnx_avx512(tag: str) -> None:
    path = f"{tag}_onnx_avx512"

    print(f"")
    print(f"正在导出 {path} ...")
    shutil.rmtree(f"{path}", ignore_errors = True)
    shutil.copytree(tag, f"{path}", dirs_exist_ok = True)
    os.remove(f"{path}/model.safetensors") if os.path.exists(f"{path}/model.safetensors") else None
    os.remove(f"{path}/pytorch_model.bin") if os.path.exists(f"{path}/pytorch_model.bin") else None

    subprocess.run(
        f"optimum-cli onnxruntime quantize --avx512 --per_channel --onnx_model {tag}_onnx -o {path}",
        shell = True,
        check = True,
    )

# 运行主函数
def main(tag) -> None:
    export_bnb_4bit(tag)
    export_bnb_8bit(tag)

    export_onnx(tag)
    export_onnx_avx512(tag)

# 运行主函数
if __name__ == "__main__":
    main(sys.argv[1])