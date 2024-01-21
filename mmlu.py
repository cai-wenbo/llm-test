import torch
from torch.utils import data
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import pandas as pd
import numpy as np
import os
import csv


def load_model(model_dir):
    model = LlamaForCausalLM.from_pretrained(model_dir,
                                             device_map="balanced",
                                             torch_dtype=torch.float16
                                             ).eval()
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def get_ans(prompt, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    logits = model(prompt_tokens.input_ids.to(device), attention_mask = prompt_tokens.attention_mask.to(device)).logits[0,-1]

    print(logits.shape)


class PromptLizer():
    def __init__(self, template):
        self.template = template

    #  def __call__(self, record):



#  def load_dataset(


if __name__ == "__main__":
    model_dir = "./models/llama-2-7b/model"
    #  model, tokenizer = load_model(model_dir)

    data_dir = "./dataset"
    subject = "machine_learning"
    data_path = os.path.join(data_dir, "dev", subject + "_dev.csv")
    df =  pd.read_csv(data_path, header = None)
    print(df.shape)
    print(df.head())
