import os
import random
import unicodedata

from tqdm import tqdm

from moudle.Normalizer import Normalizer
from moudle.TextHelper import TextHelper

PATHS = (
    "/mnt/e/ai/dataset/pt/zh",
)

# 计算字符串的实际显示长度
def get_display_lenght(text: str) -> int:
    # unicodedata.east_asian_width(c) 返回字符 c 的东亚洲宽度属性。
    # NaH 表示窄（Narrow）、中立（Neutral）和半宽（Halfwidth）字符，这些字符通常被认为是半角字符。
    # 其他字符（如全宽字符）的宽度属性为 W 或 F，这些字符被认为是全角字符。
    return sum(1 if unicodedata.east_asian_width(c) in "NaH" else 2 for c in text)

def main() -> None:
    result = []

    # 遍历所有路径
    lines = {}
    for path in PATHS:
        for file in tqdm([file for file in os.scandir(path) if file.name.endswith(".txt")]):
            with open(file.path, "r", encoding = "utf-8") as reader:
                for line in reader:
                    line = Normalizer.normalize(line, merge_space = True)
                    length = get_display_lenght(line)

                    if line == "":
                        continue

                    if "http" in line or "www" in line:
                        continue

                    if len(line) >= 2 and TextHelper.is_punctuation(line[0]) == True and TextHelper.is_punctuation(line[1]) == True:
                        continue

                    if "zh" in path and TextHelper.has_any_cjk(line) == False:
                        continue
                    elif "en" in path and TextHelper.has_any_latin(line) == False:
                        continue
                    elif "ko" in path and TextHelper.has_any_korean(line) == False:
                        continue
                    elif "ja" in path and TextHelper.has_any_japanese(line) == False:
                        continue

                    if 48 <= length < 64:
                        lines.setdefault("1", []).append(line)
                    elif 64 <= length < 96:
                        lines.setdefault("2", []).append(line)
                    elif 96 <= length < 128:
                        lines.setdefault("3", []).append(line)
                    elif 128 <= length < 192:
                        lines.setdefault("4", []).append(line)
                    elif 192 <= length < 256:
                        lines.setdefault("5", []).append(line)

    # 随机取样
    for _, v in lines.items():
        result.extend(random.sample(v, min(10000, len(v))))

    # 去重
    result = list(set(result))

    # 写入本地
    with open(f"sample_{len(result)}.txt", "w", encoding = "utf-8") as writer:
        writer.write("\n".join(result))

if __name__ == "__main__":
    main()