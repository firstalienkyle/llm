#remember to add special tokens

'''
import os
from datasets import load_dataset,DownloadConfig
import sys

hf_token = "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"
config = DownloadConfig(token=hf_token)

segment = load_dataset('HuggingFaceFW/fineweb-edu', name='default', split='train', streaming=True, download_config=config)

current_segment = iter(segment)

text = next(current_segment)['text']
'''

text = "dewihuihudwhuidqwno dsciobhu cdsio uh dsvio us fo iuas dfigou iogu sogiu dsav iogu adsf odsva "

# not yet implemented
special_tokens = ["<|ENDOFTEXT|>","<|UNK|>"]
    
def encode(text):
    return list(text.encode())

pars = {}

common_pars = {}

def common_par(text):
    for i in range(len(text)-1):
        if (text[i],text[i+1]) in pars:
            pars[(text[i],text[i+1])] +=1
        else:
            pars[(text[i],text[i+1])] = 1

    common_pars[max(pars, key=pars.get)] = len(common_pars)+256

def merge(text):
    not_merged = True
    while not_merged:
        not_merged = False
        for i in range(len(text)):
            if (text[i-1], text[i]) in common_pars:
                text[i-1:i+1] = [common_pars[(text[i-1], text[i])]] 
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

def embed():
    print("hi")

def train(text,target_vocab):
    temp_text = encode(text)
    for i in range(target_vocab-256):
        common_par(temp_text)
        temp_text = merge(temp_text)

print(len(encode(text)))
train(text,265)
print(common_pars)
print(len(merge(encode(text))))

'''
del current_segment
del segment
'''
