import pandas as pd
import spacy
from collections import Counter

spacy_en=spacy.load('en_core_web_sm')
spacy_fr=spacy.load('fr_core_news_sm')

def tokenizer_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenizer_fr(text):
    return [tok.text.lower() for tok in spacy_fr.tokenizer(text)]

def build_vocab(sentence,tokenizer,min_freq=2):
    counter=Counter()
    for sent in sentence:
        counter.update(tokenizer(sent))
    vocab={'<pad>':0,'<sos>':1,'<eos>':2,'<unk>':3}
    index=len(vocab)
    for token,freq in counter.items():
        if freq>min_freq and token not in vocab:
            vocab[token]=index
            index+=1
    return vocab

def decode(sentence,vocab,tokenizer,max_length=100):
    tokens=['<sos>']+tokenizer(sentence)+['<eos>']
    tokens_id=[vocab.get(tok,vocab['<unk>']) for tok in tokens]
    if len(tokens_id)<max_length:
        tokens_id+=[vocab['<pad>']]*(max_length-len(tokens_id))
    else:
        tokens_id=tokens_id[:max_length]
    return tokens_id

def idsToSentence(tokens_id,vocab):
    inv_vocab={v:k for k,v in vocab.items()}
    words=[]
    for idx in tokens_id.tolist():
        word=inv_vocab.get(idx,"<unk>")
        if word=='<eos>':
            break
        elif word not in ["<sos>","<pad>"]:
            words.append(word)
    return " ".join(words)



