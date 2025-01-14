import re
import copy
import json
import argparse
import traceback
import unicodedata

from tqdm import tqdm
from rich import print

# 安全加载 JSON 字典
def safe_load_json_dict(json_str: str) -> dict:
    result = {}

    # 移除首尾空白符（含空格、制表符、换行符）
    json_str = json_str.strip()

    # 移除代码标识
    json_str = json_str.removeprefix("```json").removeprefix("```").strip()

    # 先尝试使用 json.loads 解析
    try:
        result = json.loads(json_str)
    except Exception:
        pass

    # 否则使用正则表达式匹配
    if len(result) == 0:
        for item in re.findall(r"['\"].+?['\"]\s*\:\s*['\"].+?['\"]\s*(?=[,}])", json_str, flags = re.IGNORECASE):
            p = item.split(":")
            result[p[0].strip().strip("'\"").strip()] = p[1].strip().strip("'\"").strip()

    return result

# 安全加载 JSON 列表
def safe_load_json_list(json_str: str) -> list:
    result = []

    # 移除首尾空白符（含空格、制表符、换行符）
    json_str = json_str.strip()

    # 移除代码标识
    json_str = json_str.removeprefix("```json").removeprefix("```").strip()

    # 先尝试使用 json.loads 解析
    try:
        result = json.loads(json_str)
    except Exception:
        pass

    # 否则使用正则表达式匹配
    if len(result) == 0:
        for item in re.findall(r"\{.+?\}", json_str, flags = re.IGNORECASE):
            result.append(safe_load_json_dict(item))

    return result

# 计算字符串的实际显示长度
def get_display_lenght(text: str) -> int:
    # unicodedata.east_asian_width(c) 返回字符 c 的东亚洲宽度属性。
    # NaH 表示窄（Narrow）、中立（Neutral）和半宽（Halfwidth）字符，这些字符通常被认为是半角字符。
    # 其他字符（如全宽字符）的宽度属性为 W 或 F，这些字符被认为是全角字符。
    return sum(1 if unicodedata.east_asian_width(c) in "NaH" else 2 for c in text)

# 主函数
def main(target: str) -> None:
    data = []
    with open(target, "r", encoding = "utf-8") as reader:
        input = json.load(reader)

    unsupport = set()
    for item in tqdm(input, desc = target):
        if item.get("request") == "" or item.get("response") == "":
            continue

        request = item.get("request")
        response = item.get("response")

        try:
            entities = safe_load_json_list(response.get("choices")[0].get("message").get("content"))
            sentences = request.get("messages")[1].get("content").splitlines()

            # 预处理
            entities_ex = []
            for entity in entities:
                surface = entity.get("surface")

                # 跳过空字符串和单个字符
                if get_display_lenght(surface) <= 2:
                    print(f"检测到空字符串和单个字符 -> {surface}")
                    continue

                entities_ex.append(entity)
            entities = entities_ex

            # 遍历句子
            for sentence in sentences:
                entities_ex = copy.deepcopy(entities)
                for i, entity in enumerate(entities_ex):
                    # 跳过不存在于句子中的实体
                    if entity.get("surface") not in sentence:
                        entities_ex[i]["surface"] = ""
                        continue

                    # 映射实体类型
                    entity_type = entity.get("entity_type")
                    if any(v in entity_type for v in ("姓名", "姓", "姓氏", "名", "名字", "昵称", "家族", "怪物", "神")):
                        entities_ex[i]["entity_type"] = "PER"
                    elif any(v in entity_type for v in ("地点", "国家")):
                        entities_ex[i]["entity_type"] = "LOC"
                    elif any(v in entity_type for v in ("组织", "組織", "学派", "学校", "派閥")):
                        entities_ex[i]["entity_type"] = "ORG"
                    elif any(v in entity_type for v in ("物品",)):
                        entities_ex[i]["entity_type"] = "PRD"
                    else:
                        unsupport.add(entity_type)
                        entities_ex[i]["surface"] = ""
                        continue

                # 删除嵌套实体
                entities_ex = [entity for entity in entities_ex if entity.get("surface") != ""]
                entities_ex = sorted(entities_ex, key = lambda x: len(x.get("surface")), reverse = True)
                for i, _ in enumerate(entities_ex):
                    for j in range(i + 1, len(entities_ex)):
                        if entities_ex[j].get("surface") != "" and entities_ex[j].get("surface") in entities_ex[i].get("surface"):
                            print(f"检测到嵌套实体 -> {entities_ex[j].get("surface")} -> {entities_ex[i].get("surface")}")
                            entities_ex[j]["surface"] = ""

                # 添加实体列表
                entities_ex = [entity for entity in entities_ex if entity.get("surface") != ""]
                if len(entities_ex) > 0:
                    data.append(
                        {
                            "sentence": sentence,
                            "entities": entities_ex,
                        }
                    )
        except Exception as e:
            print(f"{e}")
            traceback.print_exc()

    # 写入文件
    print(f"{target.replace(".log", f"_dataset_{len(data)}.json")} -> {len(data)}")
    print(f"{unsupport}")
    with open(target.replace(".log", f"_dataset_{len(data)}.json"), "w", encoding = "utf-8") as writer:
        writer.write(json.dumps(data, indent = 4, ensure_ascii = False))

# 入口函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type = str, help = "目标文件")
    args = parser.parse_args()

    main(args.target)