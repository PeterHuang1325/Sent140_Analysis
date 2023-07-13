from transformers import BertModel, BertTokenizer, DistilBertTokenizer, DistilBertModel
from torch import nn
import torch

class DistilBERT_Sent(nn.Module):

    def __init__(self, n_classes):
        super(DistilBERT_Sent, self).__init__()
        
        self.dbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, n_classes)
        self.dropout = nn.Dropout(0.2)
        

    def forward(self, input_ids, attention_mask):
        
        outs = self.dbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
            
        hidden_state = outs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        return logits
        #return torch.sigmoid(logits)
    
    
    
class BERT_Sent(nn.Module):

    def __init__(self, n_classes):
        super(BERT_Sent, self).__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        output = self.drop(outs["pooler_output"])
        return self.out(output)
        #return torch.sigmoid(self.out(output))

        
class LogitReg(nn.Module):
    def __init__(self, n_classes):
        super(LogitReg, self).__init__()
        self.linear = nn.Linear(1000, n_classes)
    
    def forward(self, inputs):
        outputs = self.linear(inputs)
        return outputs