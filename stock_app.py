import streamlit as st
import yfinance as yf
import pandas as pd  # ←【重要】これを追加しました！
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# ページ設定
st.set_page_config(
    page_title="ガチ投資予測くん",
    page_icon="📈",
    layout="wide"
)
st.title('📈 ガチ投資予測くん for FANG+')

# 銘柄の選択肢
stocks = ("NVDA", "GOOGL", "MSFT", "AMZN", "AAPL", "TSLA")
selected_stock = st.selectbox("予測したい銘柄を選んでください", stocks)

# 予測期間の設定
n_years = st.slider("過去何年分のデータを学習させますか？", 1, 5, 2)
period = n_years * 365

st.subheader(f'選択銘柄: {selected_stock}')

# データの取得（キャッシュ機能で高速化）
@st.cache_data
def load_data(ticker):
    # 最新のyfinanceの仕様に合わせてデータを取得
    data = yf.download(ticker, start="2020-01-01", end=date.today().strftime("%Y-%m-%d"))
    
    # 【修正ポイント】多重インデックス（2階建ての列名）を平らにする処理
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('データを取得中...')
data = load_data(selected_stock)
data_load_state.text('データ取得完了！')

# 生データの表示
st.subheader('直近の株価データ（末尾5件）')
st.write(data.tail())

# グラフを描画する関数
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="終値"))
    fig.layout.update(title_text=f'{selected_stock} の株価推移', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# --- ここから機械学習 (Prophet) ---
st.subheader('🤖 AIによる未来予測')
st.write("Meta社のAIモデル『Prophet』が学習中...")

# Prophet用にデータ整形
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# モデルの作成と学習
m = Prophet()
m.fit(df_train)

# 未来の日付枠を作成
n_months = st.slider("何ヶ月先まで予測しますか？", 1, 12, 6)
future = m.make_future_dataframe(periods=n_months * 30)

# 予測実行
forecast = m.predict(future)

# 予測結果の表示
st.write(f"{n_months}ヶ月後までの予測グラフ")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("黒い点 = 実際の株価")
st.write("青い線 = AIの予測値")
st.write("薄い青の帯 = 予測の振れ幅（誤差範囲）")

# 成分分析
st.subheader('📊 傾向分析')
st.write("どの曜日に上がりやすいか、全体のトレンドはどうかが分かります。")
fig2 = m.plot_components(forecast)
st.write(fig2)

# 免責事項
st.sidebar.warning("※これは学習用アプリです。実際の投資判断は自己責任で行ってください。")
