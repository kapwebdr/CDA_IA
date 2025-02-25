import os
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

CACHE_DIR = "../cache_model"
os.makedirs(CACHE_DIR, exist_ok=True)

@dataclass
class ModelConfig:
    model_type: str
    model_path: str
    cache_dir: str = CACHE_DIR
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    repeat_penalty: float = 1.15

class ModelLoader:
    @staticmethod
    def load_transformers(config: ModelConfig) -> Tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_path, cache_dir=config.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(config.model_path, cache_dir=config.cache_dir)
        return model, tokenizer

    @staticmethod
    def load_llamacpp(config: ModelConfig) -> Tuple[Any, None]:
        from llama_cpp import Llama
        model_path = os.path.join(config.cache_dir, "llama-3.2-1b-instruct.Q4_K_M.gguf")
        if not os.path.exists(model_path):
            llm = Llama.from_pretrained(
                repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
                filename="*Q4_K_M.gguf",
                cache_dir=config.cache_dir,
                verbose=False
            )
        else:
            llm = Llama(model_path=model_path)
        return llm, None

    @staticmethod
    def load_mlx(config: ModelConfig) -> Tuple[Any, Any]:
        os.environ["HF_HOME"] = config.cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(config.cache_dir, "hub")
        from mlx_lm import load
        model, tokenizer = load(config.model_path)
        return model, tokenizer

class ModelGenerator:
    @staticmethod
    def generate_transformers(model: Any, tokenizer: Any, prompt: str, config: ModelConfig) -> Tuple[str, int, int]:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_tokens = len(inputs.input_ids[0])
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repeat_penalty
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_tokens = len(outputs[0])
        return response, input_tokens, output_tokens

    @staticmethod
    def generate_llamacpp(model: Any, prompt: str, config: ModelConfig) -> Tuple[str, int, int]:
        input_tokens = model.tokenize(prompt.encode())
        n_input_tokens = len(input_tokens)
        
        response = model(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repeat_penalty=config.repeat_penalty
        )
        response = response['choices'][0]['text'].strip()
        
        output_tokens = model.tokenize(response.encode())
        n_output_tokens = len(output_tokens)
        
        return response, n_input_tokens, n_output_tokens

    @staticmethod
    def generate_mlx(model: Any, tokenizer: Any, prompt: str, config: ModelConfig) -> Tuple[str, int, int]:
        from mlx_lm import generate
        input_tokens = len(tokenizer.encode(prompt))
        output = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=config.max_tokens
        )
        output_tokens = len(tokenizer.encode(output))
        return output, input_tokens, output_tokens 