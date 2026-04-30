import torch
import torch.nn as nn
from transformers import BertModel

class BertCNNMultiTask(nn.Module):
    def __init__(self, model_name, emotion_dim=27, distortion_classes=10, mh_classes=3):
        super(BertCNNMultiTask, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden, out_channels=128, kernel_size=k)
            for k in [2, 3, 4]
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.2)
        
        # FIXED: Names must match the saved state_dict from Try.ipynb
        self.sentiment_head = nn.Linear(384, 3)
        self.emotion_head = nn.Linear(384, emotion_dim)
        self.distortion_head = nn.Linear(384, distortion_classes)
        self.mh_head = nn.Linear(384, mh_classes)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # BERT hidden states: [batch, seq_len, hidden] -> [batch, hidden, seq_len] for Conv1d
        x = outputs.last_hidden_state.permute(0, 2, 1) 
        
        conv_outs = []
        for conv in self.convs:
            feat = torch.relu(conv(x))
            feat = self.pool(feat).squeeze(-1)
            conv_outs.append(feat)
            
        features = self.dropout(torch.cat(conv_outs, dim=1))
        
        return {
            "sentiment": self.sentiment_head(features),
            "emotion": self.emotion_head(features),
            "distortion": self.distortion_head(features),
            "mh": self.mh_head(features)
        }