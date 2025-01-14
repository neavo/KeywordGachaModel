import os
import json
import random

from tqdm import tqdm
from rich import print

THRESHOLD = 1
DIR_PATH = "dataset/ner/jp"

def main() -> None:
    datas_merged = []
    entities_map = {}
    for file in os.scandir(DIR_PATH):
        if file.name.endswith(".json") and f"{os.path.split(DIR_PATH)[1]}_" not in file.name:
            with open(file.path, "r", encoding = "utf-8") as file:
                for data in json.load(file):
                    for entity in data["entities"]:
                        key = (entity.get("name"), entity.get("ner_type"))
                        if key not in entities_map:
                            entities_map[key] = [data]
                        else:
                            entities_map[key].append(data)

    for v in entities_map.values():
        if len(v) <= THRESHOLD:
            datas_merged.extend(v)
        else:
            datas_merged.extend(random.sample(v, THRESHOLD))

    # 去重
    seen = set()
    datas_unique = []
    for v in datas_merged:
        if v["sentence"] not in seen:
            seen.add(v["sentence"])
            datas_unique.append(v)

    # 写入本地
    path = f"{DIR_PATH}/{os.path.split(DIR_PATH)[1]}_{THRESHOLD}_{len(datas_unique)}.json"
    with open(path, "w", encoding = "utf-8") as file:
        file.write(json.dumps(datas_unique, indent = 4, ensure_ascii = False))
        print(f"去重完成，{len(datas_unique)} 条数据已写入 {path} ...")

# 执行主函数
if __name__ == "__main__":
    main()