import json
import requests
import numpy as np

class FastAPIEmbeddingFunction:
    def __init__(self, emb_model, emb_endpoint, max_tokens):
        self.emb_model = emb_model
        self.emb_endpoint = emb_endpoint
        self.max_tokens = int(max_tokens)

    def embed_documents(self, texts):
        return self._call_fastapi_embedding(texts)

    def embed_query(self, text):
        return self._call_fastapi_embedding([text])[0]

    def _call_fastapi_embedding(self, texts):
        try:
            payload = {
                "input": texts,
                "model": self.emb_model,
                "truncate_prompt_tokens": self.max_tokens-1,
            }
            headers = {
                "accept": "application/json",
                "Content-type": "application/json"
            }
            response = requests.post(
                self.emb_endpoint,
                data=json.dumps(payload),
                headers=headers
            )
            response.raise_for_status()
            r = response.json()
            embeddings = [data['embedding'] for data in r['data']]
            return [np.array(embed, dtype=np.float32) for embed in embeddings]
        except requests.exceptions.RequestException as e:
            print(f"Error in _call_fastapi_embedding: {e}, {e.response.text}")
            raise e
        except Exception as e:
            print(f"Error in _call_fastapi_embedding: {e}")
            raise e
