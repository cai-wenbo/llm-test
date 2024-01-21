import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

#  from models.Baichuan2-7B.modeling_baichuan import BaichuanForCausalLM
#  from models.Baichuan2-7B.tokenization_baichuan import BaichuanTokenizer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    """
    load model and tokenizer
    """
    """
    LLama
    """
    #  model_dir = "./models/llama-2-7b/model"
    #  model_dir = "./models/llama-2-13b/model"
    #  model = LlamaForCausalLM.from_pretrained(model_dir,
    #                                           device_map="balanced",
    #                                           torch_dtype=torch.float16,
    #                                           max_memory={0: "16GB", 1: "16GB"}
    #                                           ).eval()
    #  tokenizer = LlamaTokenizer.from_pretrained(model_dir)


    
    """
    Qwen
    """
    #  model_dir = "./models/Qwen-7B"
    #  model_dir = "./models/Qwen-14B"
    #  model = AutoModelForCausalLM.from_pretrained(model_dir,
    #                                              device_map="auto",
    #                                              torch_dtype=torch.float16,
    #                                              trust_remote_code = True
    #                                              ).eval()
    #  tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code = True)



    """
    Baichuan
    """
    model_dir = "./models/Baichuan2-7B"
    model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                device_map="balanced",
                                                torch_dtype=torch.float16,
                                                trust_remote_code = True
                                                ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code = True)


    #  model = BaichuanForCausalLM.from_pretrained(model_dir)
    #  tokenizer = BaichuanTokenizer.from_pretrained(model_dir)





    
    #  model_generation_config =

    """
    Llama
    """
    prompt = input("enter your input:")
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    print(prompt_tokens.input_ids)
    output_ids = model.generate(prompt_tokens.input_ids.to(device), do_sample = True, top_k =50, temperature = 1.0, top_p = 0.9, max_length = 300)
    print(tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    while True:
        prompt = input("enter your input:")
        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        output_ids = model.generate(prompt_tokens.input_ids.to(device), do_sample = True, top_k =50, temperature = 1.0, top_p = 0.9, max_length = 300)
        print(tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
