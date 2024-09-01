import os

from tqdm import tqdm
from rich import print

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

DIR_PATH = "dataset/pretrain/kr"

def main():
    datas = []

    total = len([f for f in os.scandir(DIR_PATH) if f.name.endswith(".epub")])
    for f in tqdm(os.scandir(DIR_PATH), total = total):
        if f.name.endswith(".epub"):
            try:
                book = epub.read_epub(f.path)

                text = ""
                for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                    text = text + BeautifulSoup(item.get_content(), "html.parser").get_text()

                datas.append({
                    "name" : f.name.replace(".epub", ".txt"),
                    "text" : text
                })
            except Exception as e:
                print(f"{e}")

    for data in tqdm(datas):
        with open(f"{DIR_PATH}/{data.get("name")}", "w", encoding = "utf-8") as f:
            f.write(data.get("text"))

if __name__ == "__main__":
    main()