from flask import Flask, jsonify, request, Response
import logging
import os

from common.log_utils import set_log_level
from common.llm_utils import query_vllm_server, get_llm_model_endpoints

app = Flask(__name__)

# Globals to be set dynamically
llm_model_dict = {}

def initialize_models():
    global llm_model_dict
    llm_model_dict = get_llm_model_endpoints()

@app.post("/v1/chat/completions")
def chat_completion():
    data = request.get_json()
    if data and len(data.get("messages", [])) == 0:
        return jsonify({"error": "messages can't be empty"})
    msgs = data.get("messages")[0]
    prompt = msgs.get("content")
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.0)
    try:
        llm_model = llm_model_dict['llm_model']
        llm_endpoint = llm_model_dict['llm_endpoint']
        resp = query_vllm_server(
            prompt, llm_model, temperature, max_tokens, llm_endpoint
        )
    except Exception as e:
        return jsonify({"error": repr(e)})

    return Response(resp,
                    content_type='application/text', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Headers': 'Content-Type'
        })

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    initialize_models()

    port = int(os.getenv("PORT", "5001"))

    log_level = logging.INFO
    level = os.getenv("LOG_LEVEL", "").removeprefix("--").lower()
    if level != "":
        if "debug" in level:
            log_level == logging.DEBUG
        elif not "info" in level:
            raise Exception(f"Unknown LOG_LEVEL passed: '{level}'")
    set_log_level(log_level)

    app.run(host="0.0.0.0", port=port)
