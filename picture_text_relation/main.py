import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
# import picture_text_relation.loader as loader
from model import MyModel
from utils import seed_worker
from utils import seed_everything, train, evaluate
import loader
from sklearn.metrics import f1_score  # 添加这行导入

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--dataset', type=str, default='relationship')
parser.add_argument('--encoder_t', type=str, default='bert-base-uncased')
parser.add_argument('--encoder_v', type=str, default='resnet101', choices=['resnet101', 'resnet152'])
parser.add_argument('--stacked', action='store_true', default=False)
parser.add_argument('--rnn',   action='store_true',  default=False)
parser.add_argument('--crf',   action='store_true',  default=False)
parser.add_argument('--aux',   action='store_true',  default=False)
parser.add_argument('--gate',   action='store_true',  default=False)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'AdamW'])
args = parser.parse_args()
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

if __name__ == '__main__':
    save_dir = f'log/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)  # 添加这行
    # 将所有主要逻辑移到这个条件语句下
    seed_everything(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    # 只加载图像-文本关系分类数据集
    itr_corpus = loader.load_itr_corpus('picture_text_relation/TRC_data')
    print(f"训练集大小: {len(itr_corpus.train)}")
    print(f"测试集大小: {len(itr_corpus.test)}")

    itr_train_loader = DataLoader(itr_corpus.train, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers,
                                  shuffle=True, worker_init_fn=seed_worker, generator=generator)
    itr_test_loader = DataLoader(itr_corpus.test, batch_size=args.bs, collate_fn=list, num_workers=args.num_workers)

    model = MyModel.from_pretrained(args)
    model = model.to(device)
    # 调整优化器参数
    params = [
        {'params': model.encoder_t.parameters(), 'lr': args.lr},
        {'params': model.encoder_v.parameters(), 'lr': args.lr},
        {'params': model.proj.parameters(), 'lr': args.lr * 100},
        {'params': model.aux_head.parameters(), 'lr': args.lr * 100},
    ]
    optimizer = getattr(torch.optim, args.optim)(params)

    print(args)
    itr_losses = []
    best_test_accuracy = 0
    best_test_f1 = 0
    for epoch in range(1, args.num_epochs + 1):
        itr_loss = train(itr_train_loader, model, optimizer, task='itr', device=device)
        itr_losses.append(itr_loss)
        print(f'图像-文本关系分类在第{epoch}轮的损失: {itr_loss:.2f}')

        test_accuracy, test_f1 = evaluate(model, itr_test_loader, device=device)
        print(f'测试集上的准确率: {test_accuracy:.4f}, F1分数: {test_f1:.4f}')
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            # 保存最佳模型
            model_path = f'{save_dir}/best_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f'保存最佳模型到: {model_path}')
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1

    print(f'\n最佳测试准确率: {best_test_accuracy:.4f}')
    print(f'最佳测试F1分数: {best_test_f1:.4f}')
    results = {
        'config': vars(args),
        'itr_losses': itr_losses,
        'best_test_accuracy': best_test_accuracy,
        'best_test_f1': best_test_f1,
    }
    # 保存结果
    file_name = f'{save_dir}/bs{args.bs}_lr{args.lr}_seed{args.seed}.json'
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)
