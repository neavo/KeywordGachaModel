import os
import json

from tqdm import tqdm
from rich import print

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# 参数
PATH = "dataset/pt/ko_web"

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

        lines = ""
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            line = BeautifulSoup(item.get_content(), "html.parser").get_text().strip()
            if line != "":
                lines = lines + "\n" + line

        # 创建输出文件夹
        os.makedirs(f"{root}/output/", exist_ok = True)

        # 写入文件
        with open(f"{root}/output/{file}".replace(".epub", ".txt"), "w", encoding = "utf-8") as writer:
            writer.write(lines.strip())
    except Exception as e:
        print(f"{e}")

# 从 json 文件加载数据
def load_from_json_file(root: str, file: str) -> None:
    try:
        inputs = {}

        # 读取文件
        with open(f"{root}/{file}", "r", encoding = "utf-8") as reader:
            inputs = json.load(reader)

        # 创建输出文件夹
        os.makedirs(f"{root}/output/", exist_ok = True)

        # 写入文件
        with open(f"{root}/output/{file}".replace(".json", ".txt"), "w", encoding = "utf-8") as writer:
            writer.write("\n".join([v.strip() for v in inputs.values() if v.strip() != ""]))
    except Exception as e:
        print(f"{e}")

# 主函数
def main() -> None:
    load_from_file(PATH)

# 运行主函数
if __name__ == "__main__":
    main()