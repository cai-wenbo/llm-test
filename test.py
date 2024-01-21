import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer



model_dir = "./models/llama-2-7b"

model = LlamaForCausalLM.from_pretrained(model_dir)

tokenizer = LlamaTokenizer.from_pretrained(model_dir)
