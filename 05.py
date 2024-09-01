import os
import re
import json
import random
import asyncio
from datetime import datetime

from tqdm import tqdm
from rich import print

from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter

# 自定义参数
MODEL = "internlm2_5_20b_chat_q4_k_m"
API_KEY = "sk-no-key-required"
BASE_URL = "http://localhost:8080/v1"

BATCH = 16
TIMEOUT = 180
CHUNK_SIZE = 8 * 1024

TOP_P = 1.0
TEMPERATURE = 0.0
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0

names = set()
semaphore = asyncio.Semaphore(BATCH)
async_limiter = AsyncLimiter(max_rate = BATCH, time_period = 1)
openai_handler = AsyncOpenAI(
    api_key = API_KEY,
    base_url = BASE_URL,
    timeout = TIMEOUT,
    max_retries = 0
)

PROMPT = {}
for f in os.scandir(f"prompt"):
    if f.name.startswith("llm_ner"):
        with open(f"prompt/{f.name}", "r", encoding = "utf-8") as file:
            PROMPT[f.name.replace(".txt", "")] = file.read().strip()

DIR_PATHS = [
    "dataset/ner/zh",
    "dataset/ner/jp",
    "dataset/ner/en",
    "dataset/ner/kr",
]

NER_TYPES = [
    "PER",
    "ORG",
    "LOC",
    "PRD",
    "EVT",
]

# 分割数组
def split(datas, size):
    return [datas[i:(i + size)] for i in range(0, len(datas), size)]

def append_data(data, datas):
    i = -1
    for k, v in enumerate(datas):
        if data["sentence"] == v["sentence"]:
            i = k
            break

    if i != -1:
        datas[i]["entities"].extend(data["entities"])
    else:
        datas.append(data)


async def request(ner_type, line):
    async with semaphore, async_limiter:
        completion = await openai_handler.chat.completions.create(
            model = MODEL,
            temperature = TEMPERATURE,
            top_p = TOP_P,
            max_tokens = 256,
            presence_penalty = PRESENCE_PENALTY,
            frequency_penalty = FREQUENCY_PENALTY,
            messages = [
                {
                    "role": "user", 
                    "content": PROMPT.get(f"llm_ner_{ner_type.lower()}").replace("{line}", line)
                },
            ],
        )

        message = completion.choices[0].message
        content = message.content.replace("输出：", "")
        content = re.sub(r"\s+", "", content)

        data = {}
        data["entities"] = []
        data["sentence"] = line
        if "none" in content.lower():
            pass
        else:
            for entity in content.split(","):
                if entity == "":
                    continue

                if entity not in line:
                    continue
                
                data["entities"].append({
                    "name": entity.strip(),
                    "ner_type": ner_type,
                })

        # print(f"{data}")
        return data

def on_task_done(future, datas, tasks_successed, chunks_successed, ner_type, dir_path, chunk_length, chunks_length):
    try:
        data = future.result()

        if len(data["entities"]) > 0:
            append_data(data, datas)

        tasks_successed.append(0)
    except Exception as e:
        print(e)
    finally:
        print(f"正在处理 {dir_path} | {len(chunks_successed)} / {chunks_length} | {ner_type} | {len(tasks_successed)} / {chunk_length} ...")

def load_lines(dir_path):
    lines = []

    total = len([f for f in os.scandir(dir_path) if f.name.endswith(".txt")])
    for f in tqdm(os.scandir(dir_path), total = total):
        if f.name.endswith(".txt"):
            with open(f"{dir_path}/{f.name}", "r", encoding = "utf-8") as f:
                if "zh" in dir_path:
                    threshold = 32

                if "en" in dir_path:
                    threshold = 64

                if "jp" in dir_path:
                    threshold = 32

                if "kr" in dir_path:
                    threshold = 64
                
                lines.extend([line.strip() for line in f.readlines() if not "�" in line and len(line.strip()) > threshold])

    return lines

async def main():
    for dir_path in DIR_PATHS:
        print(f"")
        lines = load_lines(dir_path)
        lines = random.sample(lines, 8 * CHUNK_SIZE)
        print(f"")
        print(f"从 {dir_path} 加载 {len(lines)} 行文本 ...")

        # 依次处理每一种类型
        datas = []
        chunks = split(lines, CHUNK_SIZE)
        chunks_successed = []
        for i, chunk in enumerate(chunks):
            chunks_successed.append(0)
            for ner_type in NER_TYPES:
                tasks = []
                tasks_successed = []
                for line in chunk:
                    task = asyncio.create_task(request(ner_type, line))
                    
                    task.add_done_callback(
                        lambda future: on_task_done(future, datas, tasks_successed, chunks_successed, ner_type, dir_path, len(chunk), len(chunks))
                    )
                    tasks.append(task)

                # 等待异步任务完成 
                await asyncio.gather(*tasks, return_exceptions = True)

            # 写入本地
            path = f"{dir_path}/00_ner_output_{(i + 1):02d}.json"
            with open(path, "w", encoding = "utf-8") as file:
                file.write(json.dumps(datas, indent = 4, ensure_ascii = False))
                print(f"{len(datas)} 条数据已写入 {path} ...")

# 入口函数
if __name__ == "__main__":
    asyncio.run(main())