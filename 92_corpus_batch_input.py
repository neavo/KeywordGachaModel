import json

PATH = "/mnt/e/ai/dataset/ner/ko/20250121/sample_49936.txt"
LANGUAGE = "ko"

# 列表切割
def split(datas: list[str], size: int) -> list[list[str]]:
    return [datas[i:i + size] for i in range(0, len(datas), size)]

# 主函数
def main() -> None:
    with open("prompt/llm_recognize.txt", "r", encoding = "utf-8") as reader:
        prompt_llm_recognize = reader.read().strip()

    with open(PATH, "r", encoding = "utf-8") as reader:
        lines = [line.strip() for line in reader.readlines() if line.strip() != ""]

    datas = []
    for i, lines in enumerate(split(lines, 10)):
        datas.append({
            "custom_id": str(i),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "deepseek-ai/DeepSeek-R1",
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 8 * 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_llm_recognize + "\n" +"\n".join(lines),
                    },
                ],
            },
        })

    with open(PATH.replace(".txt", f"_{LANGUAGE}_batch_input.jsonl"), "w", encoding = "utf-8") as writer:
        for data in datas:
            writer.write(json.dumps(data, indent = None, ensure_ascii = False))
            writer.write("\n")


# 入口函数
if __name__ == "__main__":
    main()