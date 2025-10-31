import json
import requests
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from common.misc_utils import get_prompts

llm_classify, table_summary, query_vllm_pmt, query_vllm_stream_pmt, gen_qa_pairs_pmt = get_prompts()


def classify_text_with_llm(text_blocks, gen_model, llm_endpoint, batch_size=128):
    all_prompts = [llm_classify.format(text=item.strip()) for item in text_blocks]
    
    decisions = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Classifying Text with LLM"):
        batch_prompts = all_prompts[i:i + batch_size]

        payload = {
            "model": gen_model,
            "prompt": batch_prompts,
            "temperature": 0,
            "max_tokens": 3,
        }
        try:
            response = requests.post(f"{llm_endpoint}/v1/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            choices = result.get("choices", [])
            for choice in choices:
                reply = choice.get("text", "").strip().lower()
                decisions.append("yes" in reply)
        except requests.exceptions.RequestException as e:
            print(f"Error in vLLM: {e}, {e.response.text}")
            decisions.append(True)
        except Exception as e:
            print(f"Error in vLLM: {e}")
            decisions.append(True)
    return decisions


def filter_with_llm(text_blocks, gen_model, llm_endpoint):
    text_contents = [block.get('text') for block in text_blocks]

    # Run classification
    decisions = classify_text_with_llm(text_contents, gen_model, llm_endpoint)
    print(f"[Debug] Prompts: {len(text_contents)}, Decisions: {len(decisions)}")
    filtered_blocks = [block for dcsn, block in zip(decisions, text_blocks) if dcsn]
    print(f"[Debug] Filtered Blocks: {len(filtered_blocks)}, True Decisions: {sum(decisions)}")
    return filtered_blocks


def summarize_single_table(prompt, gen_model, llm_endpoint):
    payload = {
        "model": gen_model,
        "prompt": prompt,
        "temperature": 0,
        "repetition_penalty": 1.1,
        "max_tokens": 512,
        "stream": False,
    }
    try:
        response = requests.post(f"{llm_endpoint}/v1/completions", json=payload)
        response.raise_for_status()
        result = response.json()
        reply = result.get("choices", [{}])[0].get("text", "").strip()
        return reply
    except requests.exceptions.RequestException as e:
        print(f"Error summarizing table: {e}, {e.response.text}")
        return "No summary."
    except Exception as e:
        print(f"Error summarizing table: {e}")
        return "No summary."


def summarize_table(table_html, table_caption, gen_model, llm_endpoint, max_workers=32):
    all_prompts = [table_summary.format(content=html) for html in table_html]

    summaries = [None] * len(all_prompts)

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(all_prompts)))) as executor:
        futures = {
            executor.submit(summarize_single_table, prompt, gen_model, llm_endpoint): idx
            for idx, prompt in enumerate(all_prompts)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing Tables"):
            idx = futures[future]
            try:
                summaries[idx] = future.result()
            except Exception as e:
                print(f"Thread failed at index {idx}: {e}")
                summaries[idx] = "No summary."

    return summaries


def query_vllm(question, documents, llm_endpoint, ckpt, stop_words, max_new_tokens, stream=False, max_input_length=6000, dynamic_chunk_truncation=True):
    template_token_count=250
    context = "\n\n".join([doc.get("page_content") for doc in documents])
    
    print(f'Original Context: {context}')
    if dynamic_chunk_truncation:
        question_token_count=len(tokenize_with_llm(question, llm_endpoint))
        remaining_tokens=max_input_length-(template_token_count+question_token_count)
        context=detokenize_with_llm(tokenize_with_llm(context, llm_endpoint)[:remaining_tokens], llm_endpoint)
        print(f"Truncated Context: {context}")

    prompt = query_vllm_pmt.format(context=context, question=question)
    print("PROMPT:  ", prompt)
    headers = {
        "accept": "application/json",
        "Content-type": "application/json"
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": ckpt,
        "max_tokens": max_new_tokens,
        "repetition_penalty": 1.1,
        "temperature": 0.0,
        "stop": stop_words,
        "stream": stream
    }
    
    try:
        start_time = time.time()
        # Use requests for synchronous HTTP requests
        response = requests.post(f"{llm_endpoint}/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        end_time = time.time()
        request_time = end_time - start_time
        return response_data, request_time
    except requests.exceptions.RequestException as e:
        return {"error": str(e) + "\n" + e.response.text}, 0.
    except Exception as e:
        return {"error": str(e)}, 0.


def query_vllm_stream(question, documents, llm_endpoint, ckpt, stop_words, max_new_tokens, stream=False,
                max_input_length=6000, dynamic_chunk_truncation=True):
    template_token_count = 250
    context = "\n\n".join([doc.get("page_content") for doc in documents])

    print(f'Original Context: {context}')
    if dynamic_chunk_truncation:
        question_token_count = len(tokenize_with_llm(question, llm_endpoint))
        reamining_tokens = max_input_length - (template_token_count + question_token_count)
        context = detokenize_with_llm(tokenize_with_llm(context, llm_endpoint)[:reamining_tokens], llm_endpoint)
        print(f"Truncated Context: {context}")

    prompt = query_vllm_stream_pmt.format(context=context, question=question)
    print("PROMPT:  ", prompt)
    headers = {
        "accept": "application/json",
        "Content-type": "application/json"
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": ckpt,
        "max_tokens": max_new_tokens,
        "repetition_penalty": 1.1,
        "temperature": 0.0,
        "stop": stop_words,
        "stream": stream
    }

    try:
        # Use requests for synchronous HTTP requests
        print("STREAMING RESPONSE")
        with requests.post(f"{llm_endpoint}/v1/chat/completions", json=payload, headers=headers, stream=True) as r:
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    print("Earlier response: ", line)
                    line = line.replace("data: ", "")
                    try:
                        data = json.loads(line)
                        yield data.get("choices", [{}])[0]['delta']['content']
                    except json.JSONDecodeError:
                        print("error in decoding")
                        pass  # ignore malformed lines
    except requests.exceptions.RequestException as e:
        return {"error": str(e) + "\n" + e.response.text}, 0.
    except Exception as e:
        return {"error": str(e)}, 0.


def generate_qa_pairs(records, gen_model, gen_endpoint, batch_size=32):
    all_prompts = []
    for r in records:
        prompt = gen_qa_pairs_pmt.format(text=r.get("page_content"))
        all_prompts.append(prompt)

    qa_pairs = []

    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Generating QA Pairs"):
        batch_prompts = all_prompts[i:i+batch_size]

        payload = {
            "model": gen_model,
            "prompt": batch_prompts,
            "temperature": 0.0,
            "max_tokens": 512
        }

        try:
            response = requests.post(f"{gen_endpoint}/v1/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            choices = result.get("choices", [])

            for j, choice in enumerate(choices):
                text = choice.get("text", "").strip()
                if "Q:" in batch_prompts[j]:
                    # Try to split into question and answer
                    parts = text.split("A:", 1)
                    question = parts[0].strip().lstrip("Q:").strip()
                    answer = parts[1].strip() if len(parts) > 1 else ""
                    qa_pairs.append({
                        "question": question,
                        "answer": answer,
                        "context": records[i + j].get("page_content", ""),
                        "chunk_id": records[i + j].get("chunk_id", "")
                    })

        except requests.exceptions.RequestException as e:
            print(f"❌ Error generating QA batch: {e}, {e.response.text}")
        except Exception as e:
            print(f"❌ Error generating QA batch: {e}")

    return qa_pairs

def tokenize_with_llm(prompt, llm_endpoint):
    payload = {
        "prompt": prompt
    }
    try:
        response = requests.post(f"{llm_endpoint}/tokenize", json=payload)
        response.raise_for_status()
        result = response.json()
        tokens = result.get("tokens", [])
        return tokens
    except requests.exceptions.RequestException as e:
        print(f"Error encoding prompt: {e}, {e.response.text}")
        raise e
    except Exception as e:
        print(f"Error encoding prompt: {e}")
        raise e

def detokenize_with_llm(tokens, llm_endpoint):
    payload = {
        "tokens": tokens
    }
    try:
        response = requests.post(f"{llm_endpoint}/detokenize", json=payload)
        response.raise_for_status()
        result = response.json()
        prompt = result.get("prompt", "")
        return prompt
    except requests.exceptions.RequestException as e:
        print(f"Error decoding tokens: {e}, {e.response.text}")
        raise e
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        raise e
