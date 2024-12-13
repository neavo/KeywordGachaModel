import os
import re
import json
import random
import asyncio

from datetime import datetime

from rich import print
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter

# 设置任务参数
BATCH = 8
TIMEOUT = 180
MAX_LOOP = 32
LOOP_SIZE = 128

MODEL = "glm-4-9b-chat"
API_KEY = "sk-no-key-required"
BASE_URL = "http://localhost:8080/v1"
TOP_P = 0.95
TEMPERATURE = 0.95
PRESENCE_PENALTY = 0.95
FREQUENCY_PENALTY = 0.00

PROMPT = {}
for f in os.scandir(f"prompt"):
    if f.name.startswith("llm_corpus"):
        with open(f"prompt/{f.name}", "r", encoding = "utf-8") as file:
            PROMPT[f.name.replace(".txt", "")] = file.read().strip()

LANGUAGE = "kr"
OUTPUT_PATH = f"dataset/ner/{LANGUAGE}"
TARGET_PROMPT = PROMPT.get(f"llm_corpus_{LANGUAGE}")

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

# 异步请求
async def request():
    async with semaphore, async_limiter:
        completion = await openai_handler.chat.completions.create(
            model = MODEL,
            temperature = TEMPERATURE,
            top_p = TOP_P,
            max_tokens = 3 * 1024,
            presence_penalty = PRESENCE_PENALTY,
            frequency_penalty = FREQUENCY_PENALTY,
            messages = [
                {
                    "role": "user", 
                    "content": TARGET_PROMPT
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

        for v in result:
            v["sentence"] = v["sentence"].replace("（PER）", "")
            v["sentence"] = v["sentence"].replace("（ORG）", "")
            v["sentence"] = v["sentence"].replace("（LOC）", "")
            v["sentence"] = v["sentence"].replace("（PRD）", "")
            v["sentence"] = v["sentence"].replace("（EVT）", "")
            v["sentence"] = v["sentence"].replace("「", "")
            v["sentence"] = v["sentence"].replace("」", "")
            v["sentence"] = v["sentence"].replace("『", "")
            v["sentence"] = v["sentence"].replace("』", "")
            print(f"{v}\n")

        return result

# 异步任务完成回调
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

# 主函数
async def main():
    loop = 0
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    while loop < MAX_LOOP:
        loop = loop + 1
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
        file_path = f"{OUTPUT_PATH}/{start_time}_{MODEL.replace("/", "_").replace("-", "_")}_{loop:02d}.json"
        with open(file_path, "w", encoding = "utf-8") as file:
            file.write(json.dumps(datas, indent = 4, ensure_ascii = False))
            print(f"第 {loop} 轮已完成，数据已写入 {file_path} ...")

# 入口函数
if __name__ == "__main__":
    asyncio.run(main())