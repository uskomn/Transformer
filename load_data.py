import os
import pandas as pd

os.environ["HF_HOME"] = "./data/hf_home"
os.environ["HF_DATASETS_CACHE"] = "./data/hf_cache"
os.environ["TMPDIR"] = "./data/tmp"

from datasets import load_dataset,load_from_disk

def load_data(path,name):
    ds=load_from_disk(path)
    data=ds[name]
    print(data[0]['translation'])
    en=[item['translation']['en'] for item in data]
    fr=[item['translation']['fr'] for item in data]
    length=len(data)
    print(f"从{name}加载了{length}条数据")
    return length,en,fr

data=load_from_disk('./wmt14_fr_en_arrow/train')
n=len(data)
print(n)
print(data[0])
ds=load_data('./wmt14_fr_en_arrow','test')

