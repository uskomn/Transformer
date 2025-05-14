import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from Transformer import Transformer
from datasets import load_from_disk
from handle_data import *
from torch.optim.lr_scheduler import LambdaLR
import sacrebleu

def lr_lambda(step):
    d_model = 512
    warmup_steps = 4000
    step += 1
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)

class TranslationDataset(Dataset):
    def __init__(self,hf_data,en_vocab,fr_vocab,max_len=50):
        self.data=hf_data
        self.en_vocab=en_vocab
        self.fr_vocab=fr_vocab
        self.max_len=max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        item=self.data[idx]
        src_sentence=item['translation']['en']
        tgt_sentence=item['translation']['fr']
        src=decode(src_sentence,self.en_vocab,tokenizer_en,self.max_len)
        tgt=decode(tgt_sentence,self.fr_vocab,tokenizer_fr,self.max_len)
        return torch.tensor(src,dtype=torch.long),torch.tensor(tgt,dtype=torch.long)

# 标签平滑
class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, vocab_size, ignore_index):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.ignore_index = ignore_index
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.vocab_size = vocab_size

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = pred.data.clone().fill_(self.smoothing / (self.vocab_size - 2))
        mask = (target != self.ignore_index).unsqueeze(1)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist.masked_fill_(~mask, 0)
        return self.criterion(pred, true_dist) / mask.sum()

def evaluate_bleu(model, test_loader):
    model.eval()
    preds = []
    refs = []
    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            batch_size = src.size(0)
            outputs = torch.full((batch_size, 1), fr_vocab['<sos>'], dtype=torch.long).to(device)
            for _ in range(max_len):
                out = model(src, outputs)
                next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
                outputs = torch.cat([outputs, next_token], dim=1)
            for i in range(batch_size):
                pred_sent = idsToSentence(outputs[i], fr_vocab)
                true_sent = idsToSentence(tgt[i], fr_vocab)
                preds.append(pred_sent)
                refs.append([true_sent])  # sacrebleu expects list of references

    bleu = sacrebleu.corpus_bleu(preds, refs)
    return bleu.score

if __name__ == '__main__':
    dataset = load_from_disk('./wmt14_fr_en_arrow')
    train_data = dataset['train'].select(range(1000000))
    test_data = dataset['test']
    val_data = dataset['validation']
    print("数据加载完毕")

    train_en_sentence = [item['translation']['en'] for item in train_data]
    train_fr_sentence = [item['translation']['fr'] for item in train_data]

    en_vocab = build_vocab(train_en_sentence, tokenizer=tokenizer_en)
    fr_vocab = build_vocab(train_fr_sentence, tokenizer=tokenizer_fr)
    max_len = 50
    train_dataset = TranslationDataset(train_data, en_vocab, fr_vocab, max_len=max_len)
    val_dataset = TranslationDataset(val_data, en_vocab, fr_vocab, max_len=max_len)
    test_dataset = TranslationDataset(test_data, en_vocab, fr_vocab, max_len=max_len)
    # 放入训练主逻辑
    train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=2)
    val_loader=DataLoader(dataset=val_dataset,batch_size=32,shuffle=False,num_workers=2)
    test_loader=DataLoader(dataset=test_dataset,batch_size=32,shuffle=False,num_workers=2)

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
    criterion=LabelSmoothingLoss(0.1, len(fr_vocab), fr_vocab['<pad>'])
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # checkpoint = torch.load('./model/checkpoint.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # start_epoch = checkpoint['epoch']
    # global_step=checkpoint.get('global_step',0)

    print("开始模型训练")
    epochs=20
    start_epoch=0
    global_step=0
    for epoch in range(start_epoch,start_epoch+epochs):
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
            scheduler.step(global_step)
            global_step+=1

            total_loss+=loss.item()
        if epoch %10==0:
            ave_bleu=evaluate_bleu(model,val_loader)
            print(f"Validation Bleu: {ave_bleu:.4f}")
        print(f"Epoch {epoch+1}/{start_epoch+epochs}, Loss: {total_loss/len(train_loader):.4f}")

    ave_bleu=evaluate_bleu(model,test_loader)
    print(f"Test Bleu: {ave_bleu:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': start_epoch+epochs+1,
        'global_step':global_step
    }, './model/checkpoint.pt')

    print("Model saved")
