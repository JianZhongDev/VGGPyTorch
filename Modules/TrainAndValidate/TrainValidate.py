"""
FILENAME: TrainTest.py
DESCRIPTION: Training and testing functions
@author: Jian Zhong
"""

import torch
from .Evaluate import batch_in_top_k

# train model for one epoch
def train_one_epoch(model, train_loader, loss_func, optimizer, device):
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = 0

    model.train(True)
    for i_batch, data in enumerate(train_loader):
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        tot_nof_batch += 1

        if i_batch % 100 == 0:
            print(f"batch {i_batch} loss: {tot_loss/tot_nof_batch : >8f}")

    avg_loss = tot_loss/tot_nof_batch

    print(f"Train: Avg loss: {avg_loss:>8f}")

    return avg_loss


# basic validate model for one epoch
def validate_one_epoch(model, validate_loader, loss_func, device):
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = len(validate_loader)

    correct_samples = 0
    tot_samples = len(validate_loader.dataset)

    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(validate_loader):
            inputs, labels = data 
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            tot_loss += loss.item()
            correct_samples += (outputs.argmax(1) == labels).type(torch.float).sum().item()

    avg_loss = tot_loss/tot_nof_batch
    correct_rate = correct_samples/tot_samples

    print(f"Validate: Accuracy: {(100*correct_rate):>0.2f}%, Avg loss: {avg_loss:>8f}")

    return (avg_loss, correct_rate)


# valiate model in one epoch and return the top k-th result 
def validate_one_epoch_topk(
        model, 
        validate_loader, 
        loss_func, 
        device, 
        top_k = 1
        ):
    
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = len(validate_loader)

    correct_samples = 0
    tot_samples = len(validate_loader.dataset)

    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(validate_loader):
            inputs, labels = data 
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            tot_loss += loss.item()
            correct_samples += (batch_in_top_k(outputs, labels, top_k)).type(torch.float).sum().item()

    avg_loss = tot_loss/tot_nof_batch
    correct_rate = correct_samples/tot_samples

    print(f"Validate: top{top_k} Accuracy: {(100*correct_rate):>0.2f}%, Avg loss: {avg_loss:>8f}")

    return (avg_loss, correct_rate)


# test model in one epoch and return the top k-th result 
def validate_one_epoch_topk_aug(
        model, 
        validate_loader, 
        loss_func, 
        transforms, 
        device, 
        top_k = 1
        ):
    
    tot_loss = 0.0
    avg_loss = 0.0
    tot_nof_batch = len(validate_loader)

    correct_samples = 0
    tot_samples = len(validate_loader.dataset)

    nof_transforms = len(transforms)

    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(validate_loader):
            inputs, labels = data 
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            group_outputs = [None for _ in range(nof_transforms)]
            group_loss = [None for _ in range(nof_transforms)]
            for i_trans in range(nof_transforms):
                cur_transform = transforms[i_trans]
                cur_input = inputs
                if cur_transform is not None:
                    cur_input = cur_transform(inputs)
                cur_output = model(cur_input)
                cur_loss = loss_func(cur_output, labels)
                group_outputs[i_trans] = cur_output
                group_loss[i_trans] = cur_loss
            
            outputs = torch.mean(torch.stack(group_outputs, dim = 0), dim = 0)
            loss = torch.mean(torch.stack(group_loss, dim = 0), dim = 0)
            
            tot_loss += loss.item()
            correct_samples += (batch_in_top_k(outputs, labels, top_k)).type(torch.float).sum().item()

    avg_loss = tot_loss/tot_nof_batch
    correct_rate = correct_samples/tot_samples

    print(f"Validate: top{top_k} Accuracy: {(100*correct_rate):>0.2f}%, Avg loss: {avg_loss:>8f}")

    return (avg_loss, correct_rate)