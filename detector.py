import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# 自動下載 NLTK 必要的數據包
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class AIClassifier:
    def __init__(self):
        # AI 常用轉折詞與慣用語 (AI 密度高則扣分)
        self.ai_markers = {
            'furthermore', 'moreover', 'consequently', 'pivotal', 
            'transformative', 'fostering', 'additionally', 'comprehensive',
            'unparalleled', 'in conclusion', 'it is important to note'
        }
        
        # 人類特徵詞 (主觀、感性、口語，出現則加分)
        self.human_markers = {
            'i', 'me', 'my', 'mine', 'think', 'believe', 'feel', 
            'maybe', 'actually', 'personal', 'wonder', 'guess',
            'stuff', 'cool', 'scary', 'weird', 'honestly'
        }

    def extract_features(self, text):
        """提取文本的語言統計特徵"""
        # 清理並分詞
        sentences = sent_tokenize(text)
        words = [w.lower() for w in word_tokenize(text) if w.isalnum()]
        
        if not words or not sentences:
            return None
        
        # 1. 句長變異性 (人類寫作的核心：長短句交錯)
        sent_lengths = [len(word_tokenize(s)) for s in sentences]
        sent_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        
        # 2. 詞彙豐富度 (Type-Token Ratio)
        ttr = len(set(words)) / len(words)
        
        # 3. 詞彙密度分析
        ai_count = sum(1 for w in words if w in self.ai_markers)
        hu_count = sum(1 for w in words if w in self.human_markers)
        
        ai_density = ai_count / len(words)
        hu_density = hu_count / len(words)
        
        # 4. 句式重覆率 (以 This/It 開頭的比例)
        patterns = sum(1 for s in sentences if s.lower().startswith(('this', 'it', 'there')))
        pattern_ratio = patterns / len(sentences)

        return {
            "sent_std": sent_std,
            "ttr": ttr,
            "ai_density": ai_density,
            "hu_density": hu_density,
            "pattern_ratio": pattern_ratio,
            "word_count": len(words)
        }

    def analyze(self, text):
        f = self.extract_features(text)
        if not f:
            return {"label": "無效輸入", "confidence": 0, "features": {}, "explanation": "請輸入有效的英文文本。"}

        # --- 連續機率評分邏輯 (Scoring Logic) ---
        
        # 指標 A: 節奏感 (句長標準差越多樣越像人，18為理想波峰)
        score_rhythm = min(f['sent_std'] / 18.0, 1.0)
        
        # 指標 B: 語氣偏向 (人類詞 vs AI 詞的相對強度)
        # 基數 0.5，透過強度的差異進行連續偏移
        tone_bias = (f['hu_density'] - f['ai_density']) * 15.0 # 放大微小差異
        score_tone = max(min(0.5 + tone_bias, 1.0), 0.0)
        
        # 指標 C: 詞彙多樣性 (TTR 0.7 為人類理想上限)
        score_diversity = min(f['ttr'] / 0.7, 1.0)
        
        # 指標 D: 句式重複懲罰 (越低分越像人)
        score_pattern = 1.0 - min(f['pattern_ratio'] / 0.5, 1.0)

        # --- 最終權重融合 (Weight Fusion) ---
        # 節奏(35%) + 語氣(35%) + 多樣性(15%) + 句式(15%)
        final_human_prob = (
            (score_rhythm * 0.35) + 
            (score_tone * 0.35) + 
            (score_diversity * 0.15) + 
            (score_pattern * 0.15)
        )

        # 判定標籤與信心分數計算
        if final_human_prob >= 0.5:
            prediction = "Human"
            raw_conf = final_human_prob
        else:
            prediction = "AI"
            raw_conf = 1.0 - final_human_prob
            
        # 信心分數平滑化：讓結果分布在 51%~99% 之間，並隨特徵線性變動
        confidence = 0.51 + (raw_conf - 0.5) * 0.96
        confidence = max(min(confidence, 0.99), 0.51)

        # 生成動態解釋
        explanations = []
        if score_rhythm > 0.6:
            explanations.append("✅ **寫作節奏自然**：句子長短變化豐富，這是人類自然表達的特徵。")
        elif score_rhythm < 0.3:
            explanations.append("❌ **結構過於均勻**：句子長度缺乏波動，呈現典型的機器生成模式。")
            
        if score_tone < 0.4:
            explanations.append("❌ **語氣生硬**：偵測到大量轉折詞與中性用詞，缺乏個人主觀色彩。")
        elif score_tone > 0.6:
            explanations.append("✅ **口吻具備溫度**：使用了較多主觀感受詞彙，符合人類寫作習慣。")
            
        if f['pattern_ratio'] > 0.3:
            explanations.append("⚠️ **句式重覆度高**：過多句子以 This/It 開頭，這是 AI 為了維持邏輯常見的寫法。")

        return {
            "label": prediction,
            "confidence": confidence,
            "features": {
                "句長變異指標 (Rhythm)": f['sent_std'],
                "主觀語氣強度 (Tone)": f['hu_density'] * 100,
                "AI 特徵密度 (Marker)": f['ai_density'] * 100,
                "詞彙豐富度 (Diversity)": f['ttr']
            },
            "explanation": "\n".join(explanations) if explanations else "特徵分佈較為中性，建議增加分析字數。"
        }