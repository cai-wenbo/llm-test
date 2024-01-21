from re import template
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

    def __call__(self, record, answer = True):
        filled_prompt = self.template.format(record[0], record[1], record[2], record[3], record[4])
        if answer == True:
            filled_prompt = filled_prompt + record[5]
        return filled_prompt


class FewShot():
    def __init__(self, promptlizer, base_prompt, dev_df):
        self.promptlizer = promptlizer
        prompt = str(base_prompt)
        for idx, row in dev_df.iterrows():
            prompt = prompt + "\n" + promptlizer(row)
        self.prompt = prompt

    def __call__(self, record):
        prompt = self.prompt + "\n" + promptlizer(record, answer = False)
        return prompt
        

#  def load_dataset(


if __name__ == "__main__":
    model_dir = "./models/llama-2-7b/model"
    #  model, tokenizer = load_model(model_dir)

    template = "Question: {}\nA: {}\nB: {},\nC: {}\nD: {}\nAnswer: "
    base_prompt = "Please choose the right answer for the given questions:"
    data_dir = "./dataset"
    subject = "machine_learning"
    data_path_test = os.path.join(data_dir, "test", subject + "_test.csv")
    data_path_dev = os.path.join(data_dir, "dev", subject + "_dev.csv")
    df_dev =  pd.read_csv(data_path_dev, header = None)
    df_test =  pd.read_csv(data_path_test, header = None)

    promptlizer = PromptLizer(template)

    fewshot = FewShot(promptlizer, base_prompt, df_dev)
    r =  df.iloc[0]
    print(fewshot(r))


