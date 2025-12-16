import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.ensemble import RandomForestClassifier

# 確保 NLTK 資源下載
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class AIClassifier:
    def __init__(self):
        # 增加隨機森林的深度以提高敏感度
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.is_trained = False

    def extract_features(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        word_set = set(words)
        
        # 1. 詞彙豐富度 (AI 較低)
        ttr = len(word_set) / len(words) if len(words) > 0 else 0
        
        # 2. 平均句長 (AI 句長通常較一致且偏長)
        avg_sent_len = len(words) / len(sentences) if len(sentences) > 0 else 0
        
        # 3. 停用詞比例 (AI 常使用 fillers)
        stop_words = {'the', 'is', 'and', 'a', 'to', 'in', 'it', 'of', 'that', 'with', 'for'}
        stop_ratio = len([w for w in words if w in stop_words]) / len(words) if len(words) > 0 else 0
        
        # 4. 句子長度的標準差 (AI 句子長短通常很接近，人類則落差大)
        sent_lengths = [len(word_tokenize(s)) for s in sentences]
        sent_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

        return np.array([ttr, avg_sent_len, stop_ratio, sent_std])

    def mock_train(self):
        # 擴大模擬數據集，讓模型學會區分
        # 特徵序：[TTR, Avg_Sent_Len, Stop_Ratio, Sent_Std]
        X_train = np.array([
            # --- AI 典型的模式 (低變化、規律句長、高停用詞) ---
            [0.30, 25, 0.55, 2.0], 
            [0.35, 30, 0.50, 1.5],
            [0.32, 22, 0.48, 2.5],
            [0.38, 28, 0.52, 1.8],
            # --- Human 典型的模式 (高變化、不規律句長、低停用詞) ---
            [0.70, 12, 0.20, 12.0],
            [0.65, 15, 0.25, 10.5],
            [0.80, 10, 0.15, 15.2],
            [0.75, 18, 0.30, 8.5]
        ])
        # 0 為 AI, 1 為 Human
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def analyze(self, text):
        if not self.is_trained: self.mock_train()
        
        features = self.extract_features(text)
        prob = self.model.predict_proba(features.reshape(1, -1))[0]
        
        prediction_idx = np.argmax(prob)
        prediction = "Human" if prediction_idx == 1 else "AI"
        confidence = prob[prediction_idx]
        
        # 動態解釋
        explanations = []
        if features[0] < 0.5:
            explanations.append("- **詞彙單一性**：文字重複性偏高，符合 AI 生成特徵。")
        else:
            explanations.append("- **詞彙豐富度**：用詞變化多樣，具有人類寫作風格。")
            
        if features[3] < 5:
            explanations.append("- **節奏規律性**：句長落差極小，語氣顯得較為機械化。")
        else:
            explanations.append("- **句式變化**：長短句交錯明顯，這是人類自然寫作的特點。")
        
        return {
            "label": prediction,
            "confidence": confidence,
            "features": {
                "詞彙豐富度": features[0],
                "平均句長": features[1],
                "常用詞比例": features[2],
                "句長變化程度": features[3]
            },
            "explanation": "\n".join(explanations)
        }