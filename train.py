from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

train_losses = []
test_losses = []

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        y_pred = model(data)
        target = target.squeeze(dim=1)
        loss = criterion(y_pred, target.long())
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        processed += len(data)
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx}')
        
    train_losses.append(train_loss/processed)
        

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred = model(data)
            target = target.squeeze(dim=1)
            loss = criterion(y_pred, target.long())
            test_loss += loss.item()
            total += 1

    test_loss /= total
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))