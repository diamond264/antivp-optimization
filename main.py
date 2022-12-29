from pyrsistent import discard
import numpy as np
import torch
import argparse
import random
import sys
import os
import torch.nn.utils.prune as prune
import copy
from data_loader.data_loader import Data_loader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from collections import Counter

import conf
import models.classification

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def main():

    device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")
    print(device)

    ## log configuration
    result_path, checkpoint_path, log_path = get_path()
    if not os.path.exists(result_path):
        oldumask = os.umask(0)
        os.makedirs(result_path, 0o777)
        os.umask(oldumask)
    if not os.path.exists(checkpoint_path):
        oldumask = os.umask(0)
        os.makedirs(checkpoint_path, 0o777)
        os.umask(oldumask)

    ## hyperparameters
    if 'sst-2' in conf.args.dataset:
        opt = conf.SST2Opt

    conf.args.opt = opt
    # override learning rate if specified in arguments
    if conf.args.lr:
        opt['learning_rate'] = conf.args.lr

    ## load models
    model = None
    tokenizer = None

    if conf.args.model == 'bert':
        model = models.classification.ClassificationLM(model_path='bert-base-uncased')
    elif conf.args.model == 'mobilebert':
        model = models.classification.ClassificationLM(model_path='google/mobilebert-uncased')
    elif conf.args.model == 'bert-large':
        model = models.classification.ClassificationLM(model_path='bert-large-uncased')

    if conf.args.eval_only or conf.args.resume:
        load_checkpoint(model, conf.args.load_cp_path, device)

    ## prune model
    if conf.args.prune:
        # temp: prune every 3rd layer
        layer_to_discard = [2, 5, 8, 11]
        if conf.args.model == 'bert-large':
            layer_to_discard = [2, 5, 8, 11, 14, 17, 20, 23]
        model = layerwise_pruning(model, layer_to_discard, conf.args.discard_prob)

    ## load dataset
    tokenizer = model.get_tokenizer()
    train_dataloader, val_dataloader = Data_loader(tokenizer).get_dataloader()

    # for arg in vars(conf.args):
    #     tensorboard.log_text('args/' + arg, getattr(conf.args, arg), 0)


    #------------ Train model ------------#
    if conf.args.eval_only:
        val_loss, val_acc = evaluate_model(model, val_dataloader, device)
        print(f'val_loss: {val_loss}, val_acc: {val_acc}')
        torch.save(model.state_dict(), 'check_size.pth')
        file_stats = os.stat('check_size.pth')
        print(f'check_size: {file_stats.st_size / (1024 * 1024)} MB')
    else:
        train_model(model, train_dataloader, val_dataloader, device, checkpoint_path)
    
    return

def train_model(model, train_dataloader, val_dataloader, device, checkpoint_path):

    if conf.args.parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=conf.args.lr, eps=1e-8)

    total_steps = len(train_dataloader) * conf.args.epoch

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(conf.args.epoch):
        model.train()

        train_loss = 0.
        for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            if conf.args.test_mode:
                if idx > 100: break
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            logits = model(input_ids, attention_mask=attention_mask)

            optimizer.zero_grad()

            loss = loss_fn(logits, labels)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()

        # save model
        cp_name = 'cp_e{}.pth.tar'.format(epoch)
        save_checkpoint(model=model, epoch=epoch, checkpoint_path=checkpoint_path+cp_name)

        # validation
        val_loss ,val_acc = evaluate_model(model, val_dataloader, device)  

        print('Epoch : {}, train loss : {:.4f}, val loss : {:.4f}, val acc : {:.4f}'.format(epoch, train_loss / len(train_dataloader), val_loss, val_acc))

    pass

def evaluate_model(model, val_dataloader, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    
    val_loss = []
    val_acc = []

    model.eval()
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

            loss = loss_fn(outputs, labels)
            val_loss.append(loss.item())
            
            preds = torch.argmax(outputs, dim=1).flatten()
            acc = (preds == labels).cpu().numpy().mean() * 100
        
        val_acc.append(acc)

    val_loss = np.mean(val_loss)
    val_acc = np.mean(val_acc)
    
    return val_loss, val_acc


## This is for simple testing of model
def test_model(model, tokenizer, device):
    test_sentence = "Hello, my dog is cute"
    tokenized = tokenizer.encode_plus(
        test_sentence,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)
    
    ## Print bert embeddings
    print(outputs[0][:, 0, :])
    

def layerwise_pruning(model, layer_to_discard, discard_prob=0.8):

    # layer_to_discard = [2, 4, 6, 8]

    oldModuleList = model.backbone.encoder.layer
    newModuleList = torch.nn.ModuleList()

    for i, layer in enumerate(oldModuleList):
        if i not in layer_to_discard:
            newModuleList.append(layer)
        else: # decide whether to prune this layer
            if not random.random() < discard_prob:
                newModuleList.append(layer)

    new_model = copy.deepcopy(model)
    new_model.backbone.encoder.layer = newModuleList
    
    return new_model

def get_path():
    path = 'log/'

    # dataset, model
    path += conf.args.dataset + '/'
    path += conf.args.model + '/'

    # add log_prefix
    path += conf.args.log_prefix + '/'

    checkpoint_path = path + 'cp/'
    log_path = path
    result_path = path + '/'

    print('Path: {}'.format(path))
    return result_path, checkpoint_path, log_path

def save_checkpoint(model, epoch, checkpoint_path):

    torch.save({
        'state_dict': model.state_dict(),
        'epcoh': epoch,
    }, checkpoint_path)

def load_checkpoint(model, checkpoint_path, device):
    checkpoint_dict = torch.load(checkpoint_path, map_location=f'cuda:{conf.args.gpu_idx}')
    try:
        checkpoint = checkpoint_dict['state_dict']
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    except KeyError:
        checkpoint = checkpoint_dict
    
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()

    ## general configuration
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--log_prefix', type=str, default='', help='suffix for log file path')
    parser.add_argument('--load_cp_path', type=str, default='', help='load checkpoint in path')
    parser.add_argument('--test_mode', action='store_true', help='test mode')
    parser.add_argument('--eval_only', action='store_true', help='eval only')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--parallel', action='store_true', help='parallel training')

    ## model specific
    parser.add_argument('--model', type=str, default='mobilebert', help='model name')    
    parser.add_argument('--dataset', type=str, default='sst-2', help='dataset name')
    parser.add_argument('--freeze_bert', action='store_true', help='disable fine tuning bert')

    ## hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs')

    ## pruning specific
    parser.add_argument('--prune', action='store_true', help='prune model')
    parser.add_argument('--discard_prob', type=float, default=0.8, help='probability of discarding a layer')


    return parser.parse_args()

def set_seed():
    torch.manual_seed(conf.args.seed)
    np.random.seed(conf.args.seed)
    random.seed(conf.args.seed)
    torch.cuda.manual_seed_all(conf.args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    conf.args = parse_arguments(sys.argv[1:])
    set_seed()
    main()

