import os
import json
import argparse

import ebooklib
from tqdm import tqdm
from rich import print
from ebooklib import epub
from bs4 import BeautifulSoup

from moudle.Normalizer import Normalizer

# 加载文件
def load_from_file(path: str) -> None:
    for root, _, files in os.walk(path):
        # 从 json 文件加载数据
        for file in tqdm([file for file in files if file.endswith(".json")], desc = f"{root}"):
            load_from_json_file(root, file)

        # 从 epub 文件加载数据
        for file in tqdm([file for file in files if file.endswith(".epub")], desc = f"{root}"):
            load_from_epub_file(root, file)

# 从 epub 文件加载数据
def load_from_epub_file(root: str, file: str) -> None:
    try:
        book = epub.read_epub(f"{root}/{file}")

        # 数据处理
        lines = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            result = BeautifulSoup(item.get_content(), "html.parser").get_text().splitlines()
            for line in result:
                line = Normalizer.normalize(line, merge_space = True)
                if line != "":
                    lines.append(line)

        # 创建输出文件夹
        os.makedirs(f"{root}/output/", exist_ok = True)

        # 写入文件
        with open(f"{root}/output/{file}".replace(".epub", ".txt"), "w", encoding = "utf-8") as writer:
            writer.write("\n".join(lines))
    except Exception as e:
        print(f"{e}")

# 从 json 文件加载数据
def load_from_json_file(root: str, file: str) -> None:
    try:
        inputs = {}

        # 读取文件
        with open(f"{root}/{file}", "r", encoding = "utf-8") as reader:
            inputs = json.load(reader)

        # 数据处理
        lines = []
        for _, line in inputs.items():
            line = Normalizer.normalize(line, merge_space = True)
            if line != "":
                lines.append(line)

        # 创建输出文件夹
        os.makedirs(f"{root}/output/", exist_ok = True)

        # 写入文件
        with open(f"{root}/output/{file}".replace(".json", ".txt"), "w", encoding = "utf-8") as writer:
            writer.write("\n".join(lines))
    except Exception as e:
        print(f"{e}")

# 主函数
def main(target: str) -> None:
    load_from_file(target)

# 运行主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type = str, help = "目标路径")
    args = parser.parse_args()

    main(args.target)