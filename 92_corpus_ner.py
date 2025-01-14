import re
import json
import asyncio
import argparse
import threading

from rich import print
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter

# 设置接口
BATCH = 1
MODEL = "no"
API_KEY = "no"
BASE_URL = "http://127.0.0.1:8080"
TEMPERATURE = 0.05

# 设置任务参数
TIMEOUT = 180
CHUNK_SIZE = 10

# 线程锁
LOCK = threading.Lock()

# 限制器
SEMAPHORE = asyncio.Semaphore(BATCH)
ASYNCLIMITER = AsyncLimiter(max_rate = BATCH, time_period = 1)
OPENAICLIENT = AsyncOpenAI(
    api_key = API_KEY,
    base_url = BASE_URL,
    timeout = TIMEOUT,
    max_retries = 0
)

# 列表切割
def split(datas: list[str], size: int) -> list[list[str]]:
    return [datas[i:i + size] for i in range(0, len(datas), size)]

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

# 发起请求
async def request(lines: list[str], prompt: str, tasks: list[asyncio.Task], success: list[str], failure: list[str]) -> None:
    async with SEMAPHORE, ASYNCLIMITER:
        try:
            llm_request, llm_response, error = None, None, None

            messages = [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": "\n".join(lines),
                }
            ]

            llm_request = {
                "model" : MODEL,
                "stream" : False,
                "temperature" : TEMPERATURE,
                "max_tokens" : 4096,
                # "frequency_penalty" : 0.2 if retry == True else 0,
                "messages" : messages,
            }

            completion = await OPENAICLIENT.chat.completions.create(**llm_request)

            # OpenAI 的 API 返回的对象通常是 OpenAIObject 类型
            # 该类有一个内置方法可以将其转换为字典
            llm_response = completion.to_dict()
            usage = completion.usage
            content = completion.choices[0].message.content.strip()

            # 检查是否超过最大 token 限制
            if usage.completion_tokens >= 4096:
                raise Exception("超过最大 token 限制")

            json_list = safe_load_json_list(content)
            if len(json_list) == 0:
                raise Exception("无法解析 JSON 列表")
        except Exception as e:
            error = e
        finally:
            with LOCK:
                if error == None:
                    success.append({
                        "request": llm_request,
                        "response": llm_response,
                    })
                    print(f"成功 {len(success)} 个，失败 {len(failure)} 个，剩余 {len(tasks) - len(success) - len(failure)} 个任务 ... ")
                else:
                    failure.append({
                        "error": str(error),
                        "request": llm_request,
                        "response": llm_response,
                    })
                    print(f"成功 {len(success)} 个，失败 {len(failure)} 个，剩余 {len(tasks) - len(success) - len(failure)} 个任务 ... {str(error)}")

# 主函数
async def main(target: str) -> None:
    with open("prompt/llm_ner.txt", "r", encoding = "utf-8") as reader:
        prompt = reader.read().strip()

    with open(target, "r", encoding = "utf-8") as reader:
        lines = [line.strip() for line in reader.readlines() if line.strip() != ""]

    success = []
    failure = []

    # 切割数据
    line_chunks = split(lines, CHUNK_SIZE)

    # 执行并发任务
    tasks = []
    for lines in line_chunks:
        tasks.append(asyncio.create_task(request(lines, prompt, tasks, success, failure)))
    await asyncio.gather(*tasks, return_exceptions = True)

    # 写入成功日志
    with open(f"{target.replace(".txt", "")}_success.log", "w", encoding = "utf-8") as writer:
        writer.write(json.dumps(success, indent = 4, ensure_ascii = False))

    # 写入失败日志
    with open(f"{target.replace(".txt", "")}_failure.log", "w", encoding = "utf-8") as writer:
        writer.write(json.dumps(failure, indent = 4, ensure_ascii = False))

# 入口函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type = str, help = "目标路径")
    args = parser.parse_args()

    asyncio.run(main(args.target))