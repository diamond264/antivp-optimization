from .SST2_dataset import SST2
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.cluster import KMeans
from transformers import BertForSequenceClassification, BertModel
import torch.nn as nn
import numpy as np
import os
from collections import Counter
import pickle

import conf

class Data_loader():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_path(self):
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

    def return_dataloader(self, input, label, mask, random):
        input = torch.tensor(input)
        label = torch.tensor(label)
        mask = torch.tensor(mask)
        data = TensorDataset(input, mask, label)
        sampler = RandomSampler(data) if random else SequentialSampler(data)

        dataloader = DataLoader(data, sampler=sampler, batch_size=conf.args.batch_size if random else conf.args.batch_size, num_workers=8)
        return dataloader


    def get_dataloader(self):

        if conf.args.dataset == "sst-2":
            dataset = SST2(self.tokenizer)
        else:
            raise ValueError("dataset not found")

        print("Using dataset: ", str(dataset))

        dataset_names = ['train_inputs', 'train_masks', 'train_labels', 'val_inputs', 'val_masks', 'val_labels']
        for i in range(len(dataset_names)):
            dataset_names[i] = self.get_path()[2] + str(dataset) + '_' + dataset_names[i] + '.npy'
        
        if os.path.exists(dataset_names[0]):
            print('Loading dataset from file')
            train_inputs = np.load(dataset_names[0])
            train_masks = np.load(dataset_names[1])
            train_labels = np.load(dataset_names[2])
            val_inputs = np.load(dataset_names[3])
            val_masks = np.load(dataset_names[4])
            val_labels = np.load(dataset_names[5])

        else:
            print('Creating dataset')
            train_inputs, train_labels, train_masks, val_inputs, val_labels, val_masks = dataset.get_dataset()
            np.save(dataset_names[0], train_inputs)
            np.save(dataset_names[1], train_masks)
            np.save(dataset_names[2], train_labels)
            np.save(dataset_names[3], val_inputs)
            np.save(dataset_names[4], val_masks)
            np.save(dataset_names[5], val_labels)


        train_dataloader = self.return_dataloader(train_inputs, train_labels, train_masks, True)
        valid_dataloader = self.return_dataloader(val_inputs, val_labels, val_masks, False)
        return train_dataloader, valid_dataloader




