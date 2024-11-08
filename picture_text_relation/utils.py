import os
import random
import numpy as np
import torch
import constants
from tqdm import tqdm
# from seqeval.metrics import classification_report, f1_score
# from seqeval.scheme import IOB2
import constants
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def train(dataloader, model, optimizer, task='itr', device='cuda'):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        
        if task == 'itr':
            loss, _ = model.itr_forward(batch)
        else:
            raise ValueError(f'Unknown task: {task}')
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, list):
                labels = [pair.label for pair in batch]
            else:
                labels = batch.label
            
            _, preds = model.itr_forward(batch)
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return accuracy, f1
