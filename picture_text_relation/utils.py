import os
import random
import numpy as np
import torch
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
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


def train(train_loader, model, optimizer, task='itr'):
    losses = []

    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        loss, _ = getattr(model, f'{task}_forward')(batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def evaluate(model, test_loader):
    true_labels = []
    pred_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            _, pred = model.itr_forward(batch)
            
            # 假设每个pair只有一个整体标签
            batch_true_labels = [pair.label for pair in batch]
            
            true_labels.extend(batch_true_labels)
            pred_labels.extend(pred)

    # 确保标签是正确的格式（整数）
    true_labels = [int(label) for label in true_labels]
    pred_labels = [int(label) for label in pred_labels]

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    return accuracy, f1
