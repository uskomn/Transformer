import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Transformer import Transformer
from handle_data import *

with open('./data/train_en.csv',encoding='utf-8') as f:
    train_en=[line.strip() for line in f if line.strip()]

with open('./data/train_fr.csv',encoding='utf-8') as f:
    train_fr=[line.strip() for line in f if line.strip()]

en_train,en_test,fr_train,fr_test=train_test_split(train_en,train_fr,test_size=0.2,random_state=42)

en_vocab=build_vocab(en_train,tokenizer=tokenizer_en)
fr_vocab=build_vocab(fr_train,tokenizer=tokenizer_fr)

class TranslationDataset(Dataset):
    def __init__(self,train_en,train_fr,en_vocab,fr_vocab,max_len=50):
        self.train_en=train_en
        self.train_fr=train_fr
        self.en_vocab=en_vocab
        self.fr_vocab=fr_vocab
        self.max_len=max_len

    def __len__(self):
        return len(self.train_en)

    def __getitem__(self,idx):
        src=decode(self.train_en[idx],self.en_vocab,tokenizer_en,self.max_len)
        tgt=decode(self.train_fr[idx],self.fr_vocab,tokenizer_fr,self.max_len)
        return torch.tensor(src),torch.tensor(tgt)

max_len=50
train_dataset=TranslationDataset(en_train,fr_train,en_vocab,fr_vocab,max_len=max_len)
test_dataset=TranslationDataset(en_test,fr_test,en_vocab,fr_vocab,max_len=max_len)

train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=32,shuffle=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=Transformer(
    len(en_vocab),
    len(fr_vocab),
    d_model=512,
    N=6,
    h=8,
    d_ff=2048,
    dropout=0.1
).to(device)
criterion=nn.CrossEntropyLoss(ignore_index=fr_vocab['<pad>'])
optimizer=optim.Adam(model.parameters(),lr=0.001)

epochs=1
for epoch in range(epochs):
    model.train()
    total_loss=0
    for src,tgt in train_loader:
        src=src.to(device)
        tgt=tgt.to(device)

        tgt_index=tgt[:,:-1]
        tgt_output=tgt[:,1:]

        outputs=model(src,tgt_index)
        outputs=outputs.reshape(-1,outputs.shape[-1])
        tgt_output=tgt_output.reshape(-1)
        optimizer.zero_grad()
        loss=criterion(outputs,tgt_output)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

with torch.no_grad():
    model.eval()
    smooth=SmoothingFunction().method4
    total_bleu=0
    count=0
    for src,tgt in test_loader:
        src=src.to(device)
        batch_size=src.size(0)
        outputs=torch.full((batch_size,1),fr_vocab['<sos>'],dtype=torch.long).to(device)
        for _ in range(max_len):
            out=model(src,outputs)
            next_token=out[:,-1,:].argmax(dim=-1,keepdim=True)
            outputs=torch.cat([outputs,next_token],dim=1)
        for i in range(batch_size):
            pred_sent=idsToSentence(outputs[i],fr_vocab)
            true_sent=idsToSentence(tgt[i],fr_vocab)
            bleu=sentence_bleu(
                [true_sent.split()],pred_sent.split(),
                weights=(0.25,0.25,0.25,0.25),
                smoothing_function=smooth
            )
            total_bleu+=bleu
            count+=1
ave_bleu=total_bleu/count
print(f"Test Bleu{ave_bleu:.4f}")

torch.save(model.state_dict(),'./model/transformer.pt')
print("Model saved")