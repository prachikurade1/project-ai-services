import os
import requests
from requests.adapters import HTTPAdapter
from common.log_utils import get_logger

POOL_SIZE = 10 

adapter = HTTPAdapter(
    pool_connections=POOL_SIZE, 
    pool_maxsize=POOL_SIZE, 
    pool_block=True 
)

SESSION = requests.Session()
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)

logger = get_logger("LLM")

def get_llm_model_endpoints():
    llm_model_dict = {
        'llm_endpoint': os.getenv("LLM_ENDPOINT"),
        'llm_model':    os.getenv("LLM_MODEL"),
    }
    return llm_model_dict


def query_vllm_server(prompt, gen_model, temp, max_tokens, llm_endpoint):
    payload = {
        "model": gen_model,
        "messages": [{ "role":"system", "content": prompt }],
        "temperature": temp,
        "max_tokens": max_tokens,
    }
    try:
        response = SESSION.post(f"{llm_endpoint}/v1/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()
        reply = result.get("choices", [{}])[0].get("message", "").get("content", "")
        return reply
    except Exception as e:
        logger.error(f"Error querying vllm: {e}")
        raise e
