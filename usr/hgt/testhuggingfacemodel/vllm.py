import base64
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(base_url="http://localhost:8006/v1", api_key="EMPTY")

def file_to_data_url(path):
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def build_multimodal_content(text, image_paths):
    parts = text.split("<image>")
    content = []
    for i, seg in enumerate(parts):
        if seg.strip():
            content.append({"type": "text", "text": seg.strip()})
        if i < len(image_paths):
            content.append({"type": "image_url", "image_url": {"url": file_to_data_url(image_paths[i])}})
    return content

def infer(sample):
    user_msg = next(m for m in sample["messages"] if m["role"] == "user")
    content = build_multimodal_content(user_msg["content"], sample.get("images", []))
    resp = client.chat.completions.create(
        model="MEDEA",
        messages=[{"role": "user", "content": content}],
        temperature=0.6,
        max_tokens=4096,
        top_p=0.95,
    )
    return resp.choices[0].message.content.strip()

# Batch inference
samples = json.load(open("test_data.json"))
with ThreadPoolExecutor(max_workers=32) as ex:
    futures = {ex.submit(infer, s): i for i, s in enumerate(samples)}
    for fut in tqdm(as_completed(futures), total=len(samples)):
        i = futures[fut]
        samples[i]["pred"] = fut.result()

json.dump(samples, open("results.json", "w"), ensure_ascii=False, indent=2)
