import os
import json
import logging

LOG_LEVEL = logging.INFO

LOCAL_CACHE_DIR = os.getenv("CACHE_DIR")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

def set_log_level(level):
    global LOG_LEVEL
    LOG_LEVEL = level

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)-18s - %(levelname)-8s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

def get_prompts():
    prompts_path = os.getenv("PROMPT_PATH")

    if not prompts_path:
        raise EnvironmentError("Environment variable 'PROMPT_PATH' is not set.")

    try:
        with open(prompts_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            llm_classify = data.get("llm_classify")
            table_summary = data.get("table_summary")
            query_vllm = data.get("query_vllm")
            query_vllm_stream = data.get("query_vllm_stream")
            gen_qa_pairs = data.get("gen_qa_pairs")

            if any(prompt in (None, "") for prompt in (
                    llm_classify,
                    table_summary,
                    query_vllm,
                    query_vllm_stream,
                    gen_qa_pairs,
            )):
                raise ValueError(f"One or more prompt variables are missing or empty in '{prompts_path}' file.")

            return llm_classify, table_summary, query_vllm, query_vllm_stream, gen_qa_pairs
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found at: {prompts_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON at {prompts_path}: {e}")


def get_txt_tab_filenames(file_paths, out_path):
    original_filenames = [fp.split('/')[-1] for fp in file_paths]
    input_txt_files, input_tab_files = [], []
    for fn in original_filenames:
        f, _ = os.path.splitext(fn)
        input_txt_files.append(f'{out_path}/{f}_clean_text.json')
        input_tab_files.append(f'{out_path}/{f}_tables.json')
    return original_filenames, input_txt_files, input_tab_files


def get_model_endpoints():
    emb_model_dict = {
        'emb_endpoint': os.getenv("EMB_ENDPOINT"),
        'emb_model':    os.getenv("EMB_MODEL"),
        'max_tokens':   int(os.getenv("EMB_MAX_TOKENS")),
    }

    llm_model_dict = {
        'llm_endpoint': os.getenv("LLM_ENDPOINT"),
        'llm_model':    os.getenv("LLM_MODEL"),
    }

    return emb_model_dict, llm_model_dict
