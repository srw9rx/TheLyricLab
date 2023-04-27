import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
import os
import sys 

import random
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from transformers import AdamW
from torch.nn.utils.rnn import pad_sequence


def predict(prompt):

    #model = GPT2LMHeadModel.from_pretrained('gpt2')
    #model.load_state_dict(torch.load('rapgen.pth'))
    device = torch.device('cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|endoftext|>', eos_token='<|endoftext|>', pad_token='<|endoftext|>', padding='max_length', truncation=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    #checkpoint = torch.load('rapgen.pth', map_location=device)
    #model = model.load_state_dict(torch.load('rapgen.pth'), strict=False)
    model = torch.load('rapgenmid.pt', map_location=device)
    model.eval()
    
    #prompt = '<|startoftext|> '+ prompt
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|endoftext|>', padding=True, truncation=True)
    encodings_dict = tokenizer.encode(('<|endoftext|>'+ prompt), truncation=True, max_length=32)
    #generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = torch.tensor(encodings_dict).unsqueeze(0)
    #print(generated.size())
    generated = generated.to(device)

    #print(generated)
    #print(torch.max(generated))
    with torch.no_grad():
        sample_outputs1 = model.generate(bos_token_id=random.randint(1,30000),do_sample=True,   top_k=50,top_p=0.95, num_return_sequences=1, max_new_tokens=768)
        sample_outputs = model.generate(generated, top_k=50, top_p=0.90, num_return_sequences=1, max_new_tokens=768)
        print(sample_outputs)
        #sample_outputs = model(generated)
    #sample_outputs = model.generate(generated)
    #for i in sample_outputs1:
        #print(tokenizer.decode(i, skip_special_tokens=True))

    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True), tokenizer.decode(sample_outputs1[0], skip_special_tokens=True)
print("hi \n Prediction: ", predict("hi"))
print("harry potter \n Prediction: ", predict("harry potter"))
print("friend \n Prediction: ",predict("friend"))
print("none \n Prediction: ",predict(""))
print("hey sexy \n Prediction: ",predict("hey sexy"))