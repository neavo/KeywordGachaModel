import copy
import json
import re
import traceback
import unicodedata

from rich import print

from moudle.TextHelper import TextHelper

INPUT = "/mnt/e/ai/dataset/ner/ko/20250121/sample_49936_ko_batch_input.jsonl"
OUTPUT = "/mnt/e/ai/dataset/ner/ko/20250121/sample_49936_ko_batch_output.jsonl"

# 计算字符串的实际显示长度
def get_display_lenght(text: str) -> int:
    # unicodedata.east_asian_width(c) 返回字符 c 的东亚洲宽度属性。
    # NaH 表示窄（Narrow）、中立（Neutral）和半宽（Halfwidth）字符，这些字符通常被认为是半角字符。
    # 其他字符（如全宽字符）的宽度属性为 W 或 F，这些字符被认为是全角字符。
    return sum(1 if unicodedata.east_asian_width(c) in "NaH" else 2 for c in text)

# 主函数
def main() -> None:
    data_x = []
    with open(INPUT, "r", encoding = "utf-8") as reader:
        for line in reader:
            if line.strip() == "":
                data_x.append({})
            else:
                data_x.append(json.loads(line))

    data_y = []
    with open(OUTPUT, "r", encoding = "utf-8") as reader:
        for line in reader:
            if line.strip() == "":
                data_y.append({})
            else:
                data_y.append(json.loads(line))

    data = []
    unsupport = set()
    for x, y in zip(data_x, data_y):
        request = x.get("body")
        response = y.get("response", {}).get("body")

        if request == None or response == None:
            continue

        try:
            request_content = request.get("messages")[-1].get("content")
            sentences = re.split(r"\n\[\{.+\}\]\n", request_content)[-1].splitlines()

            response_content = response.get("choices")[0].get("message").get("content")
            entities = TextHelper.safe_load_json_list(response_content.split("</think>")[-1])

            # 跳过空字符串和单个字符
            entities = [
                entity for entity in entities
                if isinstance(entity, dict) and get_display_lenght(entity.get("surface", "")) > 2 and entity.get("entity_group", "") != ""
            ]

            # 修正错误的字段名
            for i, entity in enumerate(entities):
                if "entity_group" not in entity and "type" in entity:
                    entities[i]["entity_group"] = entities[i].get("type", "")
                elif "entity_group" not in entity and "entity_type" in entity:
                    entities[i]["entity_group"] = entities[i].get("entity_type", "")

            # 遍历句子
            for sentence in sentences:
                entities_ex = copy.deepcopy(entities)
                for i, entity in enumerate(entities_ex):
                    # 跳过不存在于句子中的实体
                    if entity.get("surface") not in sentence:
                        entities_ex[i]["surface"] = ""
                        continue

                    # 映射实体类型
                    entity_group: str = entity.get("entity_group")
                    if entity_group.lower() in ("姓名", "名字", "name", "surname", "姓氏") + ():
                        entities_ex[i]["entity_group"] = "PER"
                    elif entity_group.lower() in ("地点", "location", "建筑", "building", "地點") + ():
                        entities_ex[i]["entity_group"] = "LOC"
                    elif entity_group.lower() in ("组织", "organization", "家族", "种族", "race") + ():
                        entities_ex[i]["entity_group"] = "ORG"
                    elif entity_group.lower() in ("物品", "item", "食品", "food", "工具", "tool") + ():
                        entities_ex[i]["entity_group"] = "PRD"
                    else:
                        unsupport.add(entity_group)
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
            print(f"{e}\n{("".join(traceback.format_exception(None, e, e.__traceback__))).strip()}")

    # 写入文件
    print(f"{unsupport}")
    with open(INPUT.replace("_input.jsonl", f"_dataset_{len(data)}.json"), "w", encoding = "utf-8") as writer:
        writer.write(json.dumps(data, indent = 4, ensure_ascii = False))

# 入口函数
if __name__ == "__main__":

    main()