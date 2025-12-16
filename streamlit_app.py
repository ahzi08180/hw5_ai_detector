import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from detector import AIClassifier

st.set_page_config(page_title="AI 文章鑑定工具", layout="wide")

clf = AIClassifier()

st.title("🤖 AI vs. ✍️ Human 文章分類鑑定器")
st.markdown("透過機器學習分析文本特徵，判定文章是由 AI 生成還是人類撰寫。")

# 側邊欄：輸入方式
st.sidebar.header("輸入設定")
input_mode = st.sidebar.radio("選擇輸入方式", ["貼上文字", "上傳檔案", "範例測試"])

text_input = ""
if input_mode == "貼 on 文字":
    text_input = st.text_area("請輸入文章內容（建議至少 50 字以上）", height=300)
elif input_mode == "上傳檔案":
    uploaded_file = st.file_uploader("選擇 .txt 檔案", type=['txt'])
    if uploaded_file:
        text_input = uploaded_file.read().decode("utf-8")
else:
    text_input = "In the contemporary era of technological advancement, the integration of artificial intelligence into daily operations has become increasingly prevalent..."

if st.button("開始分析"):
    if len(text_input.strip()) < 20:
        st.warning("請輸入足夠長度的文字。")
    else:
        result = clf.analyze(text_input)
        
        # 顯示結果卡片
        col1, col2 = st.columns(2)
        with col1:
            color = "green" if result['label'] == "Human" else "red"
            st.markdown(f"### 判定結果：<span style='color:{color}'>{result['label']}</span>", unsafe_allow_html=True)
            st.metric("信心分數", f"{result['confidence']*100:.2f}%")
        
        with col2:
            st.write("#### 特徵分析說明")
            if result['features']['詞彙豐富度 (TTR)'] < 0.5:
                st.info("💡 發現特徵：詞彙重複性高，這是 AI 常見的生成模式。")
            else:
                st.info("💡 發現特徵：詞彙變化度大，較符合人類寫作習慣。")

        # 視覺化統計量
        st.divider()
        st.subheader("📊 數據統計與可視化")
        
        df_stats = pd.DataFrame({
            "特徵名稱": list(result['features'].keys()),
            "本次得分": list(result['features'].values())
        })
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.table(df_stats)
        
        with c2:
            # 畫出簡單的長條圖
            fig, ax = plt.subplots()
            sns.barplot(x="本次得分", y="特徵名稱", data=df_stats, ax=ax, palette="viridis")
            st.pyplot(fig)