import requests
from rich.console import Console
import ast
import time
import random
from openai import OpenAI

MAX_RETRIES = 12
# set your GPT url & API key here.
GPT_URL = ""
API_KEY = ""
model_name = "gpt-3.5-turbo"

def queryGPT(prompt: str, console: Console | None = None, identifier="", retry_times=12):
        # print(prompt)
    client = OpenAI(
        base_url=GPT_URL,
        # This is the default and can be omitted
        api_key=API_KEY
    )
    retry = 0
    while retry < retry_times:
        try:
            completion = client.chat.completions.create(
            messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
                timeout=15,
                temperature=0.85,
            )
            if identifier:
                if retry != 0:
                    console.log(
                        f"Task [green bold]{identifier}[/green bold] finished after {retry} retries."
                    )
                else:
                    console.log(
                        f"Task [cyan]{identifier}[/cyan] finished without retry."
                    )
            res = completion.choices[0].message.content
            break
        except:
            retry += 1
            if identifier:
                console.log(
                    f"Task [yellow]{identifier}[/yellow] retry [yellow]{retry}[/yellow] times."
                )
            time.sleep(random.uniform(0.5 + 1 * retry, 1.5 + 1 * retry))
    else:
        if identifier:
            console.log(f"Task [red]{identifier}[/red] fails. Shutdown threadpool.")
        return None
    return res


# def queryGPT(prompt: str, console: Console | None = None, identifier=""):
    assert GPT_URL and API_KEY, "You should set url and api key before queries."
    url = GPT_URL
    retry = 0
    body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.25,
    }
    headers = {
        "Content-Type": "application/json",
        "path": "/api/openai/v1/chat/completions",
        "Authorization": API_KEY,
    }
    while retry < MAX_RETRIES:
        try:
            result = requests.post(url=url, json=body, headers=headers)
            dict_response = ast.literal_eval(result.text)
            res = dict_response["choices"][0]["message"]["content"]
            if identifier:
                if retry != 0:
                    console.log(
                        f"Task [green bold]{identifier}[/green bold] finished after {retry} retries."
                    )
                else:
                    console.log(
                        f"Task [cyan]{identifier}[/cyan] finished without retry."
                    )
            break
        except:
            retry += 1
            if identifier:
                console.log(
                    f"Task [yellow]{identifier}[/yellow] retry [yellow]{retry}[/yellow] times."
                )
            time.sleep(random.uniform(0.5 + 1 * retry, 1.5 + 1 * retry))
    else:
        if identifier:
            console.log(f"Task [red]{identifier}[/red] fails. Shutdown threadpool.")
        return None
    return res


def queryLlamaCpp(
    prompt: str,
    console: Console | None = None,
    identifier: str = "",
    retry_times: int = 12,
    api_url: str = "http://localhost:8080/v1/chat/completions",
    temperature: float = 0.0,
    max_tokens: int = 300,
):
    """
    替代原 queryGPT 的本地 LLM Chat 接口，
    可直接在 DroidTask Evaluator 中使用。
    """

    # === 构建 llama.cpp 的 chat 格式 ===
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    data = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    retry = 0
    while retry < retry_times:
        try:
            res = requests.post(api_url, json=data, timeout=30)
            if res.status_code != 200:
                raise RuntimeError(f"Local LLM error: {res.text}")

            js = res.json()
            output = js["choices"][0]["message"]["content"]

            if identifier:
                if retry != 0:
                    console.log(
                        f"Task [green bold]{identifier}[/green bold] finished after {retry} retries."
                    )
                else:
                    console.log(
                        f"Task [cyan]{identifier}[/cyan] finished without retry."
                    )

            return output

        except Exception as e:
            retry += 1
            if identifier:
                console.log(
                    f"Task [yellow]{identifier}[/yellow] retry [yellow]{retry}[/yellow] times. Error: {e}"
                )
            time.sleep(random.uniform(0.5 + 1 * retry, 1.5 + 1 * retry))

    if identifier:
        console.log(f"Task [red]{identifier}[/red] fails after retries.")

    return None
