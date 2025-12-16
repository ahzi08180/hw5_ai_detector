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
        # AI 常用詞彙與轉折詞庫 (AI 傾向過度使用這些詞)
        self.ai_markers = {
            'furthermore', 'moreover', 'consequently', 'in conclusion', 
            'additionally', 'pivotal', 'crucial', 'it is important to note',
            'transformative', 'fostering', 'unparalleled', 'contemporary'
        }

    def extract_features(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        word_set = set(words)
        
        # 1. 詞彙豐富度 (TTR)
        ttr = len(word_set) / len(words) if len(words) > 0 else 0
        
        # 2. 句長變化 (標準差) - AI 的標準差通常 < 4
        sent_lengths = [len(word_tokenize(s)) for s in sentences]
        sent_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        
        # 3. AI 標記詞密度 (核心敏感度)
        marker_count = sum(1 for w in words if w in self.ai_markers)
        marker_density = marker_count / len(words) if len(words) > 0 else 0
        
        # 4. 「這 (This)」開頭句子的比例
        # AI 非常喜歡用 "This shift...", "This development..." 作為句子開頭
        this_starts = sum(1 for s in sentences if s.lower().startswith('this'))
        this_ratio = this_starts / len(sentences) if len(sentences) > 0 else 0

        return {
            "ttr": ttr,
            "sent_std": sent_std,
            "marker_density": marker_density,
            "this_ratio": this_ratio
        }

    def analyze(self, text):
        f = self.extract_features(text)
        
        # --- 高敏感度評分邏輯 ---
        # 初始分數從 0.5 開始 (0 = AI, 1 = Human)
        raw_score = 0.5
        
        # 懲罰項 (降低此分數代表趨向 AI)
        if f['sent_std'] < 5: raw_score -= 0.25  # 句長太平均
        if f['marker_density'] > 0.02: raw_score -= 0.2  # 太多 AI 轉折詞
        if f['this_ratio'] > 0.2: raw_score -= 0.15      # 過多 This 開頭句
        if f['ttr'] < 0.45: raw_score -= 0.1             # 詞彙重複
        
        # 獎勵項 (增加此分數代表趨向 Human)
        if f['sent_std'] > 10: raw_score += 0.25 # 句式落差大
        if f['ttr'] > 0.65: raw_score += 0.15    # 用詞極度豐富
        
        # 限制範圍
        final_human_prob = max(min(raw_score, 0.98), 0.02)
        
        if final_human_prob >= 0.5:
            prediction = "Human"
            confidence = final_human_prob
        else:
            prediction = "AI"
            confidence = 1.0 - final_human_prob
            
        # 解釋邏輯
        explanations = []
        if f['marker_density'] > 0.02: 
            explanations.append("- **偵測到過度使用的連接詞**：使用了大量如 Furthermore, Moreover 等 AI 偏好的轉折語。")
        if f['sent_std'] < 5: 
            explanations.append("- **語句節奏過於機械**：句子長度幾乎一致，缺乏人類寫作的隨機波動。")
        if f['this_ratio'] > 0.2:
            explanations.append("- **句式結構單一**：過多句子以「This...」開頭，這是語言模型常見的銜接方式。")

        return {
            "label": prediction,
            "confidence": confidence,
            "features": {
                "詞彙豐富度": f['ttr'],
                "句長變異性": f['sent_std'],
                "AI 特徵詞頻": f['marker_density'],
                "句式重覆率": f['this_ratio']
            },
            "explanation": "\n".join(explanations) if explanations else "- 文本特徵較為平衡，判定機率偏向中間值。"
        }