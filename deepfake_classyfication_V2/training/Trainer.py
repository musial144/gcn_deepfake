import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


class Trainer:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


    def step_batch(self, batch, criterion):
        labels = torch.stack([b["label"] for b in batch]).to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        
        logits = self.model(batch)
        loss = criterion(logits, labels)
                    
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_logits, all_labels = [], []
        
        for batch in dataloader: 
            logits = self.model(batch)
            labels = torch.stack([b["label"] for b in batch]).to(self.device)
            all_logits.append(logits)
            all_labels.append(labels)
        
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        probs = logits.softmax(dim=-1)[:,1]
        preds = logits.argmax(dim=-1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        
        try:
            auc = roc_auc_score(labels.cpu(), probs.cpu())
        except Exception:
            auc = float('nan')
        
        f1 = f1_score(labels.cpu(), preds.cpu())
        return {"acc": acc, "auc": auc, "f1": f1}