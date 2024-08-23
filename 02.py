import os
import re
import json
import random
import asyncio

from datetime import datetime

import cohere
from rich import print
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter

PROMPT_JP = (
"""
请生成用于实体识别模型训练的日文合成语料，并检查其质量。
生成时，请遵循以下内容要求、实体类别和质量标准：

内容要求：
1、生成语句：
生成10个语句，每个语句包含2-4个不同类别的实体。
每个类别的实体在每个语句中最多出现一次，以确保多样性。

2、实体使用：
每个语句中的实体词语之间不要相互包含。
实体词语应为日文片假名或平假名形式，避免使用英文或汉字。

3、符号使用：
除语法上必要的情况外，避免使用《》、「」、『』等符号包裹实体词语。

4、多样性与独特性：
语句应展现多样性，避免重复或相似的语句结构和实体。
使用随机性和不同的句子模板来增加多样性。

5、语句类型：
语句类型应包括但不限于旁白、对话、场景描述、第一人称视角、第三人称视角等。

6、题材涵盖：
语句题材应涵盖异世界、转生、穿越、奇幻、冒险、战争、科幻、历史、战国、中华风、中世纪、超能力、校园恋爱、运动竞技等轻小说常见题材。

实体类别：
1、人名（PER）：包括个体的人名，常见的名字、昵称、艺名、历史人物名字等，不包括代指人的称谓、头衔、职业和代词等。
2、组织与团体（ORG）：包括公司、机构、政府组织、非政府组织、学校、家族、门派等组织与团体。
3、地点与设施（LOC）：包括国家、城市、州、省、街道、自然地理实体（如河流、山脉）等地点或建筑物、地标、机场、桥梁、剧院、体育场等设施。
4、产品与道具（PRD）：包括物品、道具、商品、品牌、技术产品等。
5、事件（EVT）：包括历史事件、会议、发布会、庆典、比赛等。


质量标准：
1、生成的句子应具备高语言质量，确保流畅且自然。
2、各类实体在句子中的分布应合理，避免单一类型实体的过多重复。
3、确保生成的句子在日文语境中具有逻辑性和可读性，避免语法错误或不自然的表达。
4、验证实体的使用是否符合其定义，确保它们在上下文中扮演合理的角色。

回复使用JSON格式，回复中仅需要以下数据，不要出现其他文字或者描述：
[
    {
        "sentence": "<日文语句>",
        "entities": [
            {"name": "<实体名称>", "ner_type": "<PER/ORG/LOC/INS/PRD/EVT>"},
            {"name": "<实体名称>", "ner_type": "<PER/ORG/LOC/INS/PRD/EVT>"}
        ]
    }
]
"""
)

PROMPT_CN = (
"""
请生成用于实体识别模型训练的中文合成语料，并检查其质量。
生成时，请遵循以下内容要求、实体类别和质量标准：

内容要求：
1、生成语句：
生成10个语句，每个语句包含2-4个不同类别的实体。
每个类别的实体在每个语句中最多出现一次，以确保多样性。

2、实体使用：
每个语句中的实体词语之间不要相互包含。

3、符号使用：
除语法上必要的情况外，避免使用《》、「」、『』等符号包裹实体词语。

4、多样性与独特性：
语句应展现多样性，避免重复或相似的语句结构和实体。
使用随机性和不同的句子模板来增加多样性。

5、语句类型：
语句类型应包括但不限于旁白、对话、场景描述、第一人称视角、第三人称视角等。

6、题材涵盖：
语句题材应涵盖异世界、转生、穿越、奇幻、冒险、战争、科幻、历史、中世纪、超能力、校园恋爱、运动竞技等轻小说常见题材。

实体类别：
1、人名（PER）：包括个体的人名，常见的名字、昵称、艺名、历史人物名字等，不包括代指人的称谓、头衔、职业和代词等。
2、组织与团体（ORG）：包括公司、机构、政府组织、非政府组织、学校、家族、门派等组织与团体。
3、地点与设施（LOC）：包括国家、城市、州、省、街道、自然地理实体（如河流、山脉）等地点或建筑物、地标、机场、桥梁、剧院、体育场等设施。
4、产品与道具（PRD）：包括物品、道具、商品、品牌、技术产品等。
5、事件（EVT）：包括历史事件、会议、发布会、庆典、比赛等。

质量标准：
1、生成的句子应具备高语言质量，确保流畅且自然。
2、各类实体在句子中的分布应合理，避免单一类型实体的过多重复。
3、确保生成的句子应具有逻辑性和可读性，避免语法错误或不自然的表达。
4、验证实体的使用是否符合其定义，确保它们在上下文中扮演合理的角色。

回复使用JSON格式，回复中仅需要以下数据，不要出现其他文字或者描述：
[
    {
        "sentence": "<中文句子>",
        "entities": [
            {"name": "<实体名称>", "ner_type": "<PER/ORG/LOC/PRD/EVT>"},
            {"name": "<实体名称>", "ner_type": "<PER/ORG/LOC/PRD/EVT>"}
        ]
    }
]
"""
)

PROMPT_EN = (
"""
请生成用于实体识别模型训练的英文合成语料，并检查其质量。
生成时，请遵循以下内容要求、实体类别和质量标准：

内容要求：
1、生成语句：
生成10个语句，每个语句包含2-4个不同类别的实体。
每个类别的实体在每个语句中最多出现一次，以确保多样性。

2、实体使用：
每个语句中的实体词语之间不要相互包含。

3、符号使用：
除语法上必要的情况外，避免使用《》、「」、『』等符号包裹实体词语。

4、多样性与独特性：
语句应展现多样性，避免重复或相似的语句结构和实体。
使用随机性和不同的句子模板来增加多样性。

5、语句类型：
语句类型应包括但不限于旁白、对话、场景描述、第一人称视角、第三人称视角等。

6、题材涵盖：
语句题材应涵盖异世界、转生、穿越、奇幻、冒险、战争、科幻、历史、中世纪、超能力、校园恋爱、运动竞技等轻小说常见题材。

实体类别：
1、人名（PER）：包括个体的人名，常见的名字、昵称、艺名、历史人物名字等，不包括代指人的称谓、头衔、职业和代词等。
2、组织与团体（ORG）：包括公司、机构、政府组织、非政府组织、学校、家族、门派等组织与团体。
3、地点与设施（LOC）：包括国家、城市、州、省、街道、自然地理实体（如河流、山脉）等地点或建筑物、地标、机场、桥梁、剧院、体育场等设施。
4、产品与道具（PRD）：包括物品、道具、商品、品牌、技术产品等。
5、事件（EVT）：包括历史事件、会议、发布会、庆典、比赛等。

质量标准：
1、生成的句子应具备高语言质量，确保流畅且自然。
2、各类实体在句子中的分布应合理，避免单一类型实体的过多重复。
3、确保生成的句子应具有逻辑性和可读性，避免语法错误或不自然的表达。
4、验证实体的使用是否符合其定义，确保它们在上下文中扮演合理的角色。

回复使用JSON格式，回复中仅需要以下数据，不要出现其他文字或者描述：
[
    {
        "sentence": "<英文句子>",
        "entities": [
            {"name": "<实体名称>", "ner_type": "<PER/ORG/LOC/PRD/EVT>"},
            {"name": "<实体名称>", "ner_type": "<PER/ORG/LOC/PRD/EVT>"}
        ]
    }
]
"""
)

with open("02.json", "r", encoding = "utf-8") as f:
    data = json.load(f)
    MODEL = data.get("MODEL", "glm-4-9b-chat")
    API_KEY = data.get("API_KEY", "sk-no-key-required")
    BASE_URL = data.get("BASE_URL", "http://localhost:8080/v1")

BATCH = 32
TIMEOUT = 120
MAX_LOOP = 3
LOOP_SIZE = 256
TEMPERATURE = 1.25
PRESENCE_PENALTY = 1.0

PROMPT = PROMPT_CN

names = set()
semaphore = asyncio.Semaphore(BATCH)
async_limiter = AsyncLimiter(max_rate = BATCH, time_period = 1)
openai_handler = AsyncOpenAI(
    api_key = API_KEY,
    base_url = BASE_URL,
    timeout = TIMEOUT,
    max_retries = 0
)

# 使用正则表达式匹配整个模式，包括方括号
def get_string_by_rule(s):

    match = re.search(r'\[(.*?)\]', s, re.DOTALL)
    if match:
        # 返回匹配到的整个模式，包括方括号
        return match.group(0)
    else:
        # 如果没有找到匹配项，则返回空字符串或原始字符串
        return ""

# 修复不合规的JSON字符串
def fix_broken_json_string(jsonstring):
    # 在 Qwen2 7B 回复中发现
    # jsonstring = re.sub(
    #     r'(?<=: ").+(?=")', # 匹配Json字符中的值不包括双引号的部分
    #     lambda matches: matches.group(0).replace('\\"', '"').replace('"', '\\"'), 
    #     jsonstring,
    # ).strip()

    # 在 GLM4-9B 回复中发现
    jsonstring = jsonstring.replace("```json", "").replace("```", "").strip()
    # jsonstring = jsonstring.replace('“', '\\"').replace('”', '\\"').strip()
    # jsonstring = get_string_by_rule(jsonstring)
    # jsonstring = jsonstring + "}" if not jsonstring.endswith("}") else jsonstring
    # jsonstring = jsonstring.replace(",\n}", "\n}") if not jsonstring.endswith(",\n}") else jsonstring

    return jsonstring

async def request():
    async with semaphore, async_limiter:
        completion = await openai_handler.chat.completions.create(
            model = MODEL,
            temperature = TEMPERATURE,
            top_p = 0.5,
            # max_tokens = 4096,
            presence_penalty = PRESENCE_PENALTY,
            frequency_penalty = 0,
            messages = [
                {
                    "role": "user", 
                    "content": PROMPT.replace("{words}", ",".join(names))
                },
            ],
        )

        usage = completion.usage
        message = completion.choices[0].message

        json_string = message.content.strip() if message.content.strip() else "[]"
        try:
            result = json.loads(
                fix_broken_json_string(message.content.strip())
            )
        except Exception as e:
            print(message.content.strip())
            raise e

        for v1 in result:
            print(v1)
            for v2 in v1.get("entities", []):
                names.add(v2["name"])

        return result

def on_task_done(future, datas, loop, failed, successed):
    try:
        data = future.result()
        datas.extend(data)
        successed.append(0)
    except Exception as e:
        print(e)
        failed.append(0)
    finally:
        print(f"正在进行第 {loop} 轮任务，成功 {len(successed)} 次 ... 失败 {len(failed)} 次 ...")

async def main():
    loop = 0
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    while loop < MAX_LOOP:
        loop = loop + 1
        names = set()
        failed = []
        successed = []

        tasks = []
        datas = []
        for _ in range(LOOP_SIZE):
            task = asyncio.create_task(request())
            
            task.add_done_callback(lambda future: on_task_done(future, datas, loop, failed, successed))
            tasks.append(task)

        # 等待异步任务完成 
        await asyncio.gather(*tasks, return_exceptions = True)

        # 写入本地
        file_path = f"dataset\\{start_time}_{MODEL.replace("/", "_").replace("-", "_")}_{loop:02d}.json"
        with open(file_path, "w", encoding = "utf-8") as file:
            file.write(json.dumps(datas, indent = 4, ensure_ascii = False))
            print(f"第 {loop} 轮已完成，数据已写入 {file_path} ...")

if __name__ == "__main__":
    asyncio.run(main())