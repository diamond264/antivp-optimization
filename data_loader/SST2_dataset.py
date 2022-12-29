from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import conf
from tqdm import tqdm

class SST2():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __str__(self):
        return "SST2"

    def tokenizing_sst2(self, sentence):
        input_ids = []
        attention_mask = []

        print("Tokenizing SST2 data")
        for i in tqdm(range(len(sentence)), total=len(sentence)):
            sentence[i] = self.tokenizer.encode_plus(sentence[i],
                                                add_special_tokens=True,
                                                max_length=512,
                                                padding='max_length',
                                                return_attention_mask=True,
                                                return_tensors='pt')
            input_ids.append(sentence[i]['input_ids'])
            attention_mask.append(sentence[i]['attention_mask'])
            #sentence[i] = [101] + sentence[i] + [102] + ([0] * (self.cfg.max_bert_input_len - len(sentence[i])))

        return {'input_ids': torch.cat(input_ids, dim=0), 'attention_mask': torch.cat(attention_mask, dim=0)}


    def get_dataset(self):
        sst2_dataset = load_dataset("glue", "sst2")
        train_dataset = sst2_dataset["train"]
        valid_dataset = sst2_dataset["validation"]

        train_token = self.tokenizing_sst2(train_dataset["sentence"])
        valid_token = self.tokenizing_sst2(valid_dataset["sentence"])

        train_lines = train_token['input_ids'].numpy()
        valid_lines = valid_token['input_ids'].numpy()

        train_mask = train_token['attention_mask'].numpy()
        valid_mask = valid_token['attention_mask'].numpy()

        train_label = train_dataset["label"]
        valid_label = valid_dataset["label"]

        return train_lines, train_label, train_mask, valid_lines, valid_label, valid_mask



