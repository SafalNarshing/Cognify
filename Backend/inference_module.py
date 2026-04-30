import torch
from transformers import BertTokenizerFast
from BERT_CNN import BertCNNClassifier

class Config:
    model_name = "bert-base-uncased"
    mh_label_map = {'depression': 0, 'anxiety': 1, 'adhd': 2}
    rev_mh_map = {v: k for k, v in mh_label_map.items()}

cfg = Config()

def load_model(checkpoint_path="bert_cnn_best_mh.pt"):
    model = BertCNNClassifier(
        model_name=cfg.model_name,
        mh_classes=len(cfg.mh_label_map)
    )
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# Initialize components
tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
model = load_model()

def get_prediction(text: str):
    # Preprocess
    inputs = tokenizer(
        text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_token_type_ids=False
    )
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        mh_idx = torch.argmax(outputs['mh']).item()
    
    return cfg.rev_mh_map[mh_idx]