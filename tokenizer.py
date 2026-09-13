#remember to add special tokens

import os
from datasets import load_dataset,DownloadConfig
import sys

# hf_token = "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"
hf_token = input("Please enter the hf token:")
config = DownloadConfig(token=hf_token)

segment = load_dataset('HuggingFaceFW/fineweb-edu', name='default', split='train', streaming=True, download_config=config)

current_segment = iter(segment)

text = next(current_segment)['text']

# not yet implemented
special_tokens = ["<|ENDOFTEXT|>","<|UNK|>"]
    
def encode(text):
    return list(text.encode())

pars = {}

def common_par(text):
    for i in range(len(text)-1):
        if (text[i],text[i+1]) in pars:
            pars[(text[i],text[i+1])] +=1
        else:
            pars[(text[i],text[i+1])] = 1

    return max(pars, key=pars.get)

common_pars = {}

def merge(text):
    not_merged = True
    while not_merged:
        not_merged = False
        for i in range(len(text)-1):
            if (text[i],text[i+1]) in common_pars:
                text[i:i+2] = [common_pars[(text[i],text[i+1])]] 
                not_merged = True
    return text

def decode(text):
    merged = True
    new_text = []
    while merged:
        merged = False
        for i in text:
            if i in common_pars:
                new_text.append(common_pars[i])
                merged = True
            else:
                new_text.append(i)
                 
    return bytes(new_text).decode('utf-8')

def train(text,target_vocab):
    print("hi")

del current_segment
del segment
