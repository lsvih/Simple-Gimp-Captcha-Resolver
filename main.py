import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from dataloader import Dataset, test_fn, train_fn
from model import CNNCTC
from utils import load_model, decode_target, decode, characters


def main():
    if args.mode == 'train':
        train_dataset = Dataset(mode='train')
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_fn,
                                       shuffle=True, num_workers=args.workers, pin_memory=True)
        val_dataset = Dataset(mode='test')
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=test_fn,
                                     shuffle=False, num_workers=args.workers, pin_memory=True)
        train(train_loader, val_loader)
    if args.mode == 'test':
        test_dataset = Dataset(mode='test')
        test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_fn,
                                      shuffle=False, num_workers=args.workers, pin_memory=True)
        model = load_model(device)
        test(model, test_loader)


def train(train_loader, val_loader):
    max_acc = 0
    model = CNNCTC(n_classes=len(characters)).to(device)
    if args.warm_up:
        model = load_model(device)
        max_acc = float(open('warm_up_acc.txt').read())
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epoch):
        print('%d / %d Epoch' % (epoch, args.epoch))
        epoch_loss = train_epoch(train_loader, model, optimizer)
        print(epoch_loss)
        val_acc = test(model, val_loader)
        if max_acc < val_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), 'model.bin')
            open('warm_up_acc.txt', 'w').write(str(max_acc))
    return model


def train_epoch(train_loader, model, optimizer):
    total_loss = 0
    model.train()
    model.mode = 'train'
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        images = batch.images.to(device)
        labels = batch.labels.to(device)
        label_lengths = batch.label_lengths.to(device)
        probs = model(images)
        log_probs = probs.log_softmax(-1).to(device)
        prob_lengths = torch.IntTensor([log_probs.size(0)] * labels.shape[0])
        loss = F.ctc_loss(log_probs, labels, prob_lengths, label_lengths) / labels.shape[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss


def test(model, test_loader):
    model.eval()
    model.mode = 'test'
    total, correct = 0, 0
    for i, batch in enumerate(tqdm(test_loader)):
        images = batch.images.to(device)
        labels = batch.labels
        out = model(images).permute(1, 0, 2)
        for actual, label in zip(labels, out):
            label = label.argmax(dim=-1)
            label = decode(label)
            actual = decode_target(actual)
            if actual.lower() == label.lower():
                correct += 1
            total += 1
    print('acc: ', correct / total)
    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sliding convolution')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--dataset', default='gimp')
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--warm-up', default=False, type=bool)
    parser.add_argument('--lr', default=0.00002, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--mode', default='test', type=str)
    args = parser.parse_args()
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    main()
