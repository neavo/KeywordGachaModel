import json
import asyncio
import argparse
import threading

from rich import print
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter

from moudle.TextHelper import TextHelper

# 设置接口
BATCH = 16
MODEL = "no"
API_KEY = "no"
BASE_URL = "http://pc.neavo.me:8080"
TOP_P = 0.95
TEMPERATURE = 0.50

# 设置任务参数
TIMEOUT = 300
CHUNK_SIZE = 4

# 锁
LOCK_ASYNCIO = asyncio.Lock()
LOCK_THREADING = threading.Lock()

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

# 写入文件
def write(target: str, data: dict) -> None:
    with LOCK_THREADING:
        with open(target, "w", encoding = "utf-8") as writer:
            writer.write(json.dumps(data, indent = 4, ensure_ascii = False))

# 发起请求
async def request(prompt: str, content: str) -> tuple[Exception, dict, dict]:
    try:
        llm_request, llm_response, error = None, None, None

        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": content,
            }
        ]

        llm_request = {
            "model": MODEL,
            "stream": False,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": 2048,
            "messages": messages,
        }

        # 获取回复
        completion = await OPENAICLIENT.chat.completions.create(**llm_request)

        # OpenAI 的 API 返回的对象通常是 OpenAIObject 类型
        # 该类有一个内置方法可以将其转换为字典
        llm_response = completion.to_dict()
        result = TextHelper.safe_load_json_list(completion.choices[0].message.content.strip())
        if len(result) == 0:
            raise Exception("没有解析到有效 JSON 数据 ...")
    except Exception as e:
        error = e
    finally:
        return llm_request, llm_response, error

# 成功时
async def on_success(llm_request: dict, llm_response: dict, error: Exception, tasks: list[asyncio.Task], success: list[str], failure: list[str]) -> None:
    async with LOCK_ASYNCIO:
        success.append({
            "request": llm_request,
            "response": llm_response,
        })
    print(f"成功 {len(success)} 个，失败 {len(failure)} 个，剩余 {len(tasks) - len(success) - len(failure)} 个任务 ...")

# 失败时
async def on_failure(llm_request: dict, llm_response: dict, error: Exception, tasks: list[asyncio.Task], success: list[str], failure: list[str]) -> None:
    async with LOCK_ASYNCIO:
        failure.append({
            "error": str(error),
            "request": llm_request,
            "response": llm_response,
        })
    print(f"成功 {len(success)} 个，失败 {len(failure)} 个，剩余 {len(tasks) - len(success) - len(failure)} 个任务 ... {str(error)}")

# 执行任务
async def start(target: str, prompt_llm_check: str, prompt_llm_recognize: str, lines: list[str], tasks: list[asyncio.Task], success: list[str], failure: list[str]) -> None:
    async with SEMAPHORE, ASYNCLIMITER:
        # 获取 LLM 识别结果
        error = None
        llm_request, llm_response, error = await request(
            prompt_llm_recognize,
            "\n".join(lines),
        )

        if error == None:
            pass
        else:
            await on_failure(llm_request, llm_response, error, tasks, success, failure)
            return

        # 数据处理
        result = {}
        result["entities"] = TextHelper.safe_load_json_list(llm_response.get("choices")[0].get("message").get("content").strip())
        result["sentences"] = "\n".join(lines)

        # 获取 LLM 检查结果
        error = None
        llm_request, llm_response, error = await request(
            prompt_llm_check,
            json.dumps(
                result,
                indent = None,
                ensure_ascii = False,
            ),
        )

        if error == None:
            await on_success(llm_request, llm_response, error, tasks, success, failure)
        else:
            await on_failure(llm_request, llm_response, error, tasks, success, failure)

        # 写入文件
        if len(success) + len(failure) > 0 and (len(success) + len(failure)) % 5 == 0:
            write(f"{target.replace(".txt", "")}_failure.log", failure)
            write(f"{target.replace(".txt", "")}_success.log", success)

# 主函数
async def main(target: str) -> None:
    with open("prompt/llm_check.txt", "r", encoding = "utf-8") as reader:
        prompt_llm_check = reader.read().strip()

    with open("prompt/llm_recognize.txt", "r", encoding = "utf-8") as reader:
        prompt_llm_recognize = reader.read().strip()

    with open(target, "r", encoding = "utf-8") as reader:
        lines = [line.strip() for line in reader.readlines() if line.strip() != ""]

    success = []
    failure = []

    # 切割数据
    line_chunks = split(lines, CHUNK_SIZE)

    # 执行并发任务
    tasks = []
    for lines in line_chunks:
        tasks.append(asyncio.create_task(start(target, prompt_llm_check, prompt_llm_recognize, lines, tasks, success, failure)))
    await asyncio.gather(*tasks, return_exceptions = True)

    # 写入文件
    write(f"{target.replace(".txt", "")}_failure.log", failure)
    write(f"{target.replace(".txt", "")}_success.log", success)

# 入口函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type = str, help = "目标路径")
    args = parser.parse_args()

    asyncio.run(main(args.target))