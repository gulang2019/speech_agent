import urllib.request
import json

payload = {
    "model": "google/gemma-3-4b-it",
    "messages": [
        {"role": "system",
         "content": "You are a concise, helpful voice assistant. Keep answers short."},
        {
            "role": "user",
            "content": " Did you know that the average NFL game has only eleven minutes of live gameplay?"
        },
    ],
    "temperature": 0.2,
    "stream": True,
    "ignore_eos": True, 
    "max_tokens": 53
}
headers = {"Content-Type": "application/json", "Authorization": "Bearer tok-123"}
url = "http://127.0.0.1:8000/v1/chat/completions"
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    url,
    data=data,
    headers=headers,
    method="POST",
)
timeout_s = 60
with urllib.request.urlopen(req, timeout=timeout_s) as resp:
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        print(json.loads(data_str))
