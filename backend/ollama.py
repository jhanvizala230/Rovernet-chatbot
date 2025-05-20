import requests
import json
from typing import Generator

# from logger import logger

# def ask_ollama(prompt,image_base64, model="llama3.2:latest"):
#     print(f"Model : {model},Prompt : {prompt}")
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": model,
#             "prompt": prompt,
#             "image" : [image_base64],
#             "stream": False
#         }
#     )
#     return response.json()["response"].strip()

def ask_ollama_streaming(prompt: str, image_base64: str, model: str = "llava:7b") -> Generator[str, None, None]:
    """
    Stream response from Ollama with LLaVA
    
    Args:
        prompt: Text prompt
        image_base64: Base64 encoded image
        model: Model name
    
    Yields:
        Response chunks as they arrive
    """
    try:
        with requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": True
            },
            stream=True,
            timeout=30
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    if not chunk.get("done"):
                        yield chunk.get("response", "")

    except requests.exceptions.RequestException as e:
        yield f"🚨 Error: {str(e)}"


# Backward-compatible version
def ask_ollama(prompt: str, image_base64: str, model: str = "llava:7b"):
    """
    Dual-mode function that supports both streaming and single-response
    """
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
# Example usage
