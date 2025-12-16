import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class AIClassifier:
    def __init__(self):
        self.ai_markers = {'furthermore', 'moreover', 'consequently', 'pivotal', 'transformative', 'fostering', 'additionally'}
        self.human_markers = {'i', 'me', 'my', 'think', 'believe', 'feel', 'maybe', 'actually', 'personal', 'wonder'}

    def extract_features(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        if not words: return None
        
        sent_lengths = [len(word_tokenize(s)) for s in sentences]
        
        return {
            "sent_std": np.std(sent_lengths) if len(sent_lengths) > 1 else 0,
            "ai_density": sum(1 for w in words if w in self.ai_markers) / len(words),
            "human_density": sum(1 for w in words if w in self.human_markers) / len(words),
            "ttr": len(set(words)) / len(words),
            "word_count": len(words)
        }

    def analyze(self, text):
        f = self.extract_features(text)
        if not f: return {"label": "Error", "confidence": 0, "features": {}, "explanation": ""}
        
        # --- 連續機率計算邏輯 ---
        # 我們將各項指標轉換為 0~1 之間的機率值，最後取加權平均
        
        # 1. 句長變異性得分 (越多樣越像人, 15為滿分基準)
        s_std = min(f['sent_std'] / 15.0, 1.0)
        
        # 2. 詞彙豐富度得分 (0.7為人寫滿分基準)
        s_ttr = min(f['ttr'] / 0.7, 1.0)
        
        # 3. 語氣偏向得分 (人類詞彙 vs AI 詞彙)
        # 預設 0.5，人詞多就往上加，AI詞多就往下減
        tone_diff = (f['human_density'] - f['ai_density']) * 10 # 放大差異
        s_tone = max(min(0.5 + tone_diff, 1.0), 0.0)
        
        # --- 最終加權平均 ---
        # 權重分配：變異性(40%) + 語氣(40%) + 豐富度(20%)
        final_human_prob = (s_std * 0.4) + (s_tone * 0.4) + (s_ttr * 0.2)
        
        # 判定
        if final_human_prob >= 0.5:
            prediction = "Human"
            confidence = final_human_prob
        else:
            prediction = "AI"
            confidence = 1.0 - final_human_prob
            
        # 確保信心分數看起來更自然 (51%~99%)
        confidence = 0.5 + (confidence - 0.5) * 0.98
        confidence = max(min(confidence, 0.99), 0.51)

        # 解釋邏輯
        explanations = []
        if s_std > 0.7: explanations.append("✅ **節奏鮮明**：句子長短變化極大，極具人類隨性寫作特徵。")
        elif s_std < 0.3: explanations.append("❌ **節奏死板**：句子長度異常均勻，與 AI 生成邏輯吻合。")
        
        if s_tone < 0.4: explanations.append("❌ **學術腔調重**：偵測到過多 AI 偏好的轉接詞彙。")
        elif s_tone > 0.6: explanations.append("✅ **主觀口吻**：文字中帶有明顯的人類情緒或第一人稱特徵。")

        return {
            "label": prediction,
            "confidence": confidence,
            "features": {
                "句長多樣性指標": f['sent_std'],
                "人類詞彙密度": f['human_density'],
                "AI 詞彙密度": f['ai_density'],
                "詞彙豐富度": f['ttr']
            },
            "explanation": "\n".join(explanations) if explanations else "文字特徵混合，判定為中性。建議增加字數以提高準確度。"
        }