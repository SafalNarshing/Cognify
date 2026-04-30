import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast
from BERT_CNN_multi import BertCNNMultiTask

class MultiTaskInference:
    def __init__(self, model_path):
        # Mappings aligned with Try.ipynb and main-folder.ipynb
        self.sentiment_labels = ["Negative", "Neutral", "Positive"]
        self.mh_labels = ["depression", "anxiety", "adhd"] 
        self.distortion_labels = [
            'All-or-nothing thinking', 'Overgeneralization', 'Mental filter',
            'Should statements', 'Labeling', 'Personalization', 'Magnification',
            'Emotional Reasoning', 'Mind Reading', 'Fortune-telling'
        ]
        self.emotion_labels = [
            "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire",
            "disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy",
            "love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise"
        ]

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = BertCNNMultiTask("bert-base-uncased")
    
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # REMOVED strict=False. This is crucial for debugging.
        self.model.load_state_dict(state_dict) 
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 1. Sentiment
            sent_idx = torch.argmax(outputs['sentiment'], dim=1).item()
            
            # 2. Emotions (Multi-label threshold 0.3)
            em_probs = torch.sigmoid(outputs['emotion']).squeeze()
            detected_emotions = [self.emotion_labels[i] for i, p in enumerate(em_probs) if p > 0.3]
            
            # 3. Distortion
            dist_idx = torch.argmax(outputs['distortion'], dim=1).item()
            
            # 4. Mental Health
            mh_idx = torch.argmax(outputs['mh'], dim=1).item()
            
            return {
                "Sentiment": self.sentiment_labels[sent_idx],
                "Detected Emotions": ", ".join(detected_emotions) if detected_emotions else "None",
                "Cognitive Distortion": self.distortion_labels[dist_idx],
                "Mental Health Signal": self.mh_labels[mh_idx]
            }