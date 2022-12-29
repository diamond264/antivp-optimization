import torch
import torch.nn as nn

from transformers import MobileBertModel, MobileBertTokenizer, BertModel, BertTokenizer, AutoModel, AutoTokenizer
import conf

def load_backbone(name):
    model = None
    tokenizer = None

    if conf.args.model == 'bert':
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif conf.args.model == 'mobilebert':
        model = MobileBertModel.from_pretrained('google/mobilebert-uncased')
        tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
    elif conf.args.model == 'bert-large':
        model = BertModel.from_pretrained('bert-large-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    else:
        raise ValueError('No matching backbone network found')

    return model, tokenizer

class ClassificationLM(nn.Module):

    def __init__(self, model_path):
        super(ClassificationLM, self).__init__()

        self.model_path = model_path
        backbone, tokenizer = load_backbone(self.model_path)
        
        self.backbone = backbone
        self.tokenizer = tokenizer
        
        ## Freezing bert parameters
        if conf.args.freeze_bert:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.n_classes = conf.args.opt['num_class']
        self.dropout = nn.Dropout(0.1) # in BERT paper, "the dropout probability was always kept at 0.1"
        
        dim = self.get_feat_dim(conf.args.model)
        self.net_cls = nn.Linear(dim, self.n_classes) # classification head
    
    def forward(self, x, attention_mask):
        # attention_mask = (x > 0).float() # 0 is the pad_token for BERT and AlBERT

        #out_h, out_p = self.backbone(x, attention_mask, return_dict=False)
        out = self.backbone(x, attention_mask)
        out_p = out[0][:, 0, :]
        out_p = self.dropout(out_p)
        out_cls = self.net_cls(out_p)

        return out_cls
    
    def get_feature(self, x):
        attention_mask = (x > 0).float()
        out_h, out_p = self.backbone(x, attention_mask, return_dict=False)
        out_p = self.dropout(out_p)

        return out_p

    def get_tokenizer(self):
        return self.tokenizer

    def get_feat_dim(self, model_name):
        dic = {
            'bert': 768,
            'bert-tiny': 128,
            'bert-mini': 256,
            'bert-small': 512,
            'bert-medium': 512,
            'mobilebert': 512,
            'bert-large': 1024,
        }
        return dic[model_name]

