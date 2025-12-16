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
        # AI 特徵詞 (過多則扣分)
        self.ai_markers = {'furthermore', 'moreover', 'consequently', 'pivotal', 'transformative', 'fostering'}
        # 人類特徵詞 (出現則加分：個人化、感性、口語)
        self.human_markers = {'i ', 'me', 'my', 'think', 'believe', 'feel', 'maybe', 'actually', 'personal'}

    def extract_features(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # 1. 句長變異性 (人類寫作的核心：長短句交錯)
        sent_lengths = [len(word_tokenize(s)) for s in sentences]
        sent_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        
        # 2. AI 詞彙密度
        ai_density = sum(1 for w in words if w in self.ai_markers) / len(words) if len(words) > 0 else 0
        
        # 3. 人類特徵密度
        human_density = sum(1 for w in words if w in self.human_markers) / len(words) if len(words) > 0 else 0
        
        # 4. 詞彙豐富度 (TTR)
        ttr = len(set(words)) / len(words) if len(words) > 0 else 0

        return {
            "sent_std": sent_std,
            "ai_density": ai_density,
            "human_density": human_density,
            "ttr": ttr
        }

    def analyze(self, text):
        f = self.extract_features(text)
        
        # 基準分：0.5 (AI < 0.5 < Human)
        score = 0.5
        
        # --- 動態加減分邏輯 ---
        # 句長落差獎勵 (人類寫作通常大於 10)
        if f['sent_std'] > 12: score += 0.25
        elif f['sent_std'] < 4: score -= 0.2
        
        # 個人化語氣獎勵
        if f['human_density'] > 0.01: score += 0.2
        
        # AI 轉折詞懲罰
        if f['ai_density'] > 0.015: score -= 0.25
        
        # 詞彙豐富度平衡
        if f['ttr'] > 0.6: score += 0.1
        elif f['ttr'] < 0.4: score -= 0.1

        # 最終判定
        final_score = max(min(score, 0.98), 0.02)
        prediction = "Human" if final_score >= 0.5 else "AI"
        confidence = final_score if final_score >= 0.5 else 1.0 - final_score
        
        # 視覺化解釋
        explanations = []
        if f['sent_std'] > 12: explanations.append("✅ **節奏感強**：觀察到明顯的長短句交錯，符合人類自然寫作風格。")
        if f['human_density'] > 0.01: explanations.append("✅ **個人化色彩**：文中包含主觀視角詞彙，這在 AI 生成中較少見。")
        if f['ai_density'] > 0.015: explanations.append("❌ **AI 慣用轉折**：連接詞使用過於刻意，與語言模型生成習慣吻合。")
        if f['sent_std'] < 4: explanations.append("❌ **語句僵硬**：句子長度過於平均，顯得機械化。")

        return {
            "label": prediction,
            "confidence": confidence,
            "features": {
                "句長變異性 (越跳動越像人)": f['sent_std'],
                "人類特徵詞頻": f['human_density'],
                "AI 特徵詞頻": f['ai_density'],
                "詞彙豐富度": f['ttr']
            },
            "explanation": "\n".join(explanations) if explanations else "文字特徵均勻，處於判定邊界。"
        }