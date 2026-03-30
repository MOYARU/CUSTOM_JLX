import os
import requests
import json
import numpy as np

def fetch_wiki_sample():
    # Fetch a few random Wikipedia pages in Korean
    print("Fetching Wikipedia data...")
    url = "https://ko.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "CustomJLX-Bot/1.0 (rbffo@example.com)"
    }
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 10
    }
    
    response = requests.get(url, params=params, headers=headers).json()
    pages = response['query']['random']
    
    all_text = ""
    for page in pages:
        p_params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": True,
            "pageids": page['id']
        }
        p_res = requests.get(url, params=p_params, headers=headers).json()
        extract = list(p_res['query']['pages'].values())[0].get('extract', '')
        all_text += extract + "\n\n"

    
    return all_text

def main():
    text = fetch_wiki_sample()
    if not text:
        print("Failed to fetch text.")
        return
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    
    print(f"Vocab size: {vocab_size}")
    
    # Save vocab for C++ reference if needed
    with open("data/vocab.json", "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False)
    
    # Encode
    tokens = [stoi[c] for c in text]
    tokens_np = np.array(tokens, dtype=np.int32)
    
    os.makedirs("data", exist_ok=True)
    tokens_np.tofile("data/wiki_tokens.bin")
    print(f"Saved {len(tokens)} tokens to data/wiki_tokens.bin")

if __name__ == "__main__":
    main()
