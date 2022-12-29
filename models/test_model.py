from torch import nn
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        D_in, H, D_out = 768, 50, 2
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')


        # if cfg.custom_embed:
        #     self.classifier = nn.Sequential(
        #         nn.Linear(D_in, H),
        #         nn.ReLU(),
        #         nn.Linear(H, D_out)
        #     )
        # else:
        self.classifier = nn.Sequential(
            nn.Linear(D_in, D_out)
        )

        # self.maxpool = nn.MaxPool1d(cfg.num_pool, stride=cfg.num_pool)

        if freeze_bert:
            for param in self.bert_parameters():
                param.requires_grad = False

    def get_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return tokenizer

    def forward(self, input_ids, attention_mask):
        b,_ = input_ids.shape
        # if cfg.is_pooling:
        #     input_ids = torch.reshape(input_ids, (b*cfg.num_pool, cfg.max_bert_input_len))
        #     attention_mask = torch.reshape(attention_mask, (b*cfg.num_pool, cfg.max_bert_input_len))

        output = self.bert(input_ids = input_ids, attention_mask = attention_mask)

        output = output[0][:, 0, :]
        #print('embedding', output.shape)

        # if cfg.is_pooling:
        #     output = torch.permute(self.maxpool(torch.permute(output, (1,0))), (1,0))

        #print('maxpool', output.shape)

        logits = self.classifier(output)

        return logits