import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.ensemble import RandomForestClassifier

# 下載必要的 NLTK 數據
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class AIClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def extract_features(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        word_set = set(words)
        
        ttr = len(word_set) / len(words) if len(words) > 0 else 0
        avg_sent_len = len(words) / len(sentences) if len(sentences) > 0 else 0
        stop_words = {'the', 'is', 'and', 'a', 'to', 'in', 'it', 'of', 'that'}
        stop_ratio = len([w for w in words if w in stop_words]) / len(words) if len(words) > 0 else 0
        punctuation_count = len(re.findall(r'[!?]', text)) / len(sentences) if len(sentences) > 0 else 0

        return np.array([ttr, avg_sent_len, stop_ratio, punctuation_count])

    def mock_train(self):
        X_train = np.array([
            [0.35, 28, 0.50, 0.05], [0.40, 25, 0.45, 0.1], # AI 典型
            [0.75, 12, 0.20, 0.6], [0.65, 15, 0.25, 0.4]   # Human 典型
        ])
        y_train = np.array([0, 0, 1, 1])
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def analyze(self, text):
        if not self.is_trained: self.mock_train()
        features = self.extract_features(text)
        prob = self.model.predict_proba(features.reshape(1, -1))[0]
        
        prediction = "Human" if np.argmax(prob) == 1 else "AI"
        confidence = max(prob)
        
        # 根據特徵生成動態解釋
        explanations = []
        if features[0] < 0.5: explanations.append("- 詞彙變化較少，重複率高。")
        else: explanations.append("- 詞彙豐富度高，具有人類寫作特徵。")
        
        if features[1] > 20: explanations.append("- 句子平均長度較長且結構規律。")
        
        return {
            "label": prediction,
            "confidence": confidence,
            "features": {
                "詞彙豐富度": features[0],
                "平均句長": features[1],
                "常用詞比例": features[2],
                "感性符號率": features[3]
            },
            "explanation": "\n".join(explanations)
        }