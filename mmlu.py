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


def get_logits(prompt, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    logits = model(prompt_tokens.input_ids.to(device), attention_mask = prompt_tokens.attention_mask.to(device)).logits[0,-1]
    return logits



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
        

class Evaluator():
    def __init__(self, model, tokenizer, promptlizer, choices_id):
        self.model = model
        self.tokenizer = tokenizer
        self.promptlizer = promptlizer
        self.choices_id = choices_id
        self.mapping_list =  {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    """
    return the accuracy for the given df
    """

    def __call__(self, df_dev, df_test, base_prompt):
        #  get labels
        labels = df_test[5].map(self.mapping_list).values
        print(labels)
        logits_of_interest = np.zeros((df_test.shape[0], 4))
        fewshot = FewShot(self.promptlizer, base_prompt, df_dev)
        for idx, row in df_test.iterrows():
            print(idx)
            prompt = fewshot(row)
            #  get logits
            logits = get_logits(prompt, self.tokenizer, self.model).tolist()

            logits_of_interest[idx] = [logits[i] for i in self.choices_id]
        
        #  get preds
        preds = np.argmax(logits_of_interest, axis = 1)

        print(labels.shape)
        print(preds.shape)
        correct = preds == labels
        accuracy = np.mean(correct)
        
        return accuracy
    



if __name__ == "__main__":
    model_dir = "./models/llama-2-7b/model"
    data_dir = "./dataset"
    choices = ["A", "B", "C", "D"]

    
    template = "Question: {}\nA: {}\nB: {},\nC: {}\nD: {}\nAnswer: "
    base_prompt = "Please choose the most appropriate answer for the given questions, answer should be among A, B, C, D."
    
    """
    load model
    """
    model, tokenizer = load_model(model_dir)

    promptlizer = PromptLizer(template)
    choices_id = [tokenizer(choice).input_ids[-1] for choice in choices]

    evaluator = Evaluator(model, tokenizer, promptlizer, choices_id)

    accuracy_dict = dict()
    """
    get subjects
    """
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f])
    for subject in subjects:
        """
        load data
        """
        data_path_test = os.path.join(data_dir, "test", subject + "_test.csv")
        data_path_dev = os.path.join(data_dir, "dev", subject + "_dev.csv")
        df_dev =  pd.read_csv(data_path_dev, header = None)
        df_test =  pd.read_csv(data_path_test, header = None)

        accuracy = evaluator(df_dev, df_test, base_prompt)
        accuracy_dict[subject] = accuracy


    
