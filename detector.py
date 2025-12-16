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
        # 這裡仍然保留模型，但我們會結合統計權重來讓分數更靈敏
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

    def extract_features(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        word_set = set(words)
        
        # 特徵 1: 詞彙豐富度 (TTR) - 人類通常 > 0.6
        ttr = len(word_set) / len(words) if len(words) > 0 else 0
        
        # 特徵 2: 平均句長 - AI 常在 20-30 之間
        avg_sent_len = len(words) / len(sentences) if len(sentences) > 0 else 0
        
        # 特徵 3: 停用詞比例 - AI 較高
        stop_words = {'the', 'is', 'and', 'a', 'to', 'in', 'it', 'of', 'that', 'with', 'for'}
        stop_ratio = len([w for w in words if w in stop_words]) / len(words) if len(words) > 0 else 0
        
        # 特徵 4: 句長變化 (標準差) - 人類通常 > 8
        sent_lengths = [len(word_tokenize(s)) for s in sentences]
        sent_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

        return np.array([ttr, avg_sent_len, stop_ratio, sent_std])

    def mock_train(self):
        # 這裡我們模擬較極端的數據點，讓模型學習區分邊界
        X_train = np.array([
            [0.3, 30, 0.6, 1.0], [0.35, 25, 0.5, 2.0], # AI 典型
            [0.7, 10, 0.2, 15.0], [0.8, 15, 0.1, 10.0]  # Human 典型
        ])
        y_train = np.array([0, 0, 1, 1])
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def analyze(self, text):
        if not self.is_trained: self.mock_train()
        
        feat = self.extract_features(text)
        
        # --- 動態權重評分系統 (讓分數動起來的關鍵) ---
        # 我們計算一個「人類指數」(0 到 1)
        human_score = 0.0
        
        # TTR 貢獻 (越高越像人類)
        human_score += min(feat[0] / 0.8, 1.0) * 0.4 
        # 句長變化貢獻 (變化越大越像人類)
        human_score += min(feat[3] / 15.0, 1.0) * 0.4
        # 停用詞貢獻 (越低越像人類)
        human_score += (1.0 - min(feat[2] / 0.6, 1.0)) * 0.2
        
        # 結合模型預測與權重評分
        model_prob = self.model.predict_proba(feat.reshape(1, -1))[0]
        
        # 最終信心分數混合 (50% 來自模型, 50% 來自統計特徵)
        final_human_prob = (model_prob[1] * 0.5) + (human_score * 0.5)
        
        if final_human_prob >= 0.5:
            prediction = "Human"
            confidence = final_human_prob
        else:
            prediction = "AI"
            confidence = 1.0 - final_human_prob
            
        # 修正信心分數過於極端的情況
        confidence = clip_confidence = max(min(confidence, 0.99), 0.51)

        # 動態解釋
        explanations = []
        if feat[0] < 0.45: explanations.append("- **詞彙豐富度低**：用詞重複性高。")
        else: explanations.append("- **詞彙豐富度高**：用詞靈活。")
        
        if feat[3] < 5: explanations.append("- **句式過於規律**：缺乏人類寫作的隨機性。")
        else: explanations.append("- **句式變化明顯**：長短句交錯，符合人類特徵。")
        
        return {
            "label": prediction,
            "confidence": confidence,
            "features": {
                "詞彙豐富度": feat[0],
                "平均句長": feat[1],
                "常用詞比例": feat[2],
                "句長變化程度": feat[3]
            },
            "explanation": "\n".join(explanations)
        }