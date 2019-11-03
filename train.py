import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from helper import epoch_time
import time
import math

def train(model, iterator, optimizer, criterion, clip):
    model.train()    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):

        if i % 10 == 0 or i == len(iterator) - 1:
            print(f'\r{i+1:4}/{len(iterator):4} {(i+1) * 100.0 /len(iterator):.1f}%', end='')

        src = batch[0]
        trg = batch[1]
        
        optimizer.zero_grad()        
        output = model(src, trg)        
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)        
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)        
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)        
        optimizer.step()
        
        epoch_loss += loss.item()
    print() 
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch[0]
            trg = batch[1]

            output = model(src, trg, 0) #turn off teacher forcing
            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)    

def train_model(model, train_iterator, valid_iterator, optimizer, criterion, CLIP, N_EPOCHS, model_pt_path):
    best_valid_loss = float('inf')
    last_exp_v = float('inf')
    last_exp_t = float('inf')
    inc_streak_v = 0
    inc_streak_t = 0

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_pt_path)

        exp_t = int(math.exp(train_loss))
        exp_v = int(math.exp(valid_loss))
        torch.save(model.state_dict(), model_pt_path.replace('.pt', f'_E{epoch}T{exp_t}V{exp_v}.pt'))

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if exp_v > 20000:
            break 
        if exp_v > last_exp_v:
            inc_streak_v += 1
        else :
            inc_streak_v = 0
        if exp_t > last_exp_t:
            inc_streak_t += 1
        else :
            inc_streak_t = 0
        
        if inc_streak_v > 3 or inc_streak_t > 3:
            break
        last_exp_v = exp_v
        last_exp_t = exp_t
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')