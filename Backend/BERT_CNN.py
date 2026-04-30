import torch
from transformers import BertModel, BertTokenizerFast
import torch.nn as nn

class BertCNNClassifier(nn.Module):
    def __init__(self, model_name,  mh_classes, cnn_filters=128, kernel_sizes=[2,3,4], dropout=0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden, out_channels=cnn_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        concat_size = cnn_filters * len(kernel_sizes)
        
        # Task heads
        
        self.mh_head = nn.Linear(concat_size, mh_classes)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = bert_out.last_hidden_state.permute(0, 2, 1)  # [batch, hidden, seq_len]
        
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(last_hidden))
            p = self.pool(c).squeeze(-1)
            conv_outs.append(p)
            
        features = torch.cat(conv_outs, dim=1)
        features = self.dropout(features)
        
        return {
            
            'mh': self.mh_head(features)
        }