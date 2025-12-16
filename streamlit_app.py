import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from detector import AIClassifier

# 設定頁面
st.set_page_config(page_title="AI 文章鑑定工具", layout="wide")

# 初始化分類器
clf = AIClassifier()

# 定義範例文字
SAMPLE_TEXT = "In the contemporary era of technological advancement, the integration of artificial intelligence into daily operations has become increasingly prevalent. This shift offers unparalleled efficiency but also raises concerns about authenticity."

# --- UI 介面 ---
st.title("🤖 AI vs. ✍️ Human 文章分類鑑定器")
st.markdown("透過機器學習分析文本特徵，判定文章是由 AI 生成還是人類撰寫。")

# 側邊欄設定
st.sidebar.header("輸入設定")
input_mode = st.sidebar.radio("選擇輸入方式", ["貼上文字", "上傳檔案", "範例測試"])

# 根據選擇模式決定 text_input 的初始值
current_text = ""

if input_mode == "範例測試":
    current_text = SAMPLE_TEXT
elif input_mode == "上傳檔案":
    uploaded_file = st.sidebar.file_uploader("選擇 .txt 檔案", type=['txt'])
    if uploaded_file:
        current_text = uploaded_file.read().decode("utf-8")
    else:
        st.info("請在側邊欄上傳檔案")
else:
    current_text = "" # 讓使用者手動輸入

# 顯示文字輸入框（這會讓使用者看到目前的內容，也可以手動修改）
text_to_analyze = st.text_area("待分析文章內容", value=current_text, height=300, placeholder="在此輸入或修改內容...")

if st.button("開始分析"):
    if len(text_to_analyze.strip()) < 20:
        st.warning("⚠️ 請輸入足夠長度的文字（至少 20 個字元）。")
    else:
        # 執行分析
        result = clf.analyze(text_to_analyze)
        
        # --- 結果顯示區 ---
        st.divider()
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("判定結論")
            color = "#2ecc71" if result['label'] == "Human" else "#e74c3c"
            st.markdown(f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="color: white; margin: 0;">{result['label']}</h1>
                    <p style="color: white; font-size: 1.2rem; margin-top: 10px;">信心分數: {result['confidence']*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 原因分析")
            st.info(result['explanation'] if 'explanation' in result else "根據語言統計特徵進行判定。")

        with col2:
            st.subheader("📊 特徵數據可視化")
            
            # 準備數據並將 Key 轉為英文以確保繪圖不出錯
            plot_data = pd.DataFrame({
                "Feature": ["Vocabulary Richness", "Avg Sentence Length", "Stopword Ratio", "Sentence Variability"],
                "Score": list(result['features'].values())
            })
            
            # 使用 Streamlit 原生圖表 (自動避開中文字體問題)
            # 將 Feature 設為索引以利 st.bar_chart 讀取
            st.bar_chart(data=plot_data, x="Feature", y="Score", color="#4db6ac")
            
            # 下方表格保留中文，表格在網頁渲染不會有亂碼問題
            df_display = pd.DataFrame({
                "特徵指標": list(result['features'].keys()),
                "數值": [f"{v:.3f}" for v in result['features'].values()]
            })
            st.dataframe(df_display, use_container_width=True)