import streamlit as st  # pythonライブラリ
import pandas as pd  # dataの分析と操作
from pandas_datareader import data as pdr   # 金融データを読み取るツール
import datetime  # 日付と時刻　
import yfinance as yf  # Yahoo Financeから金融データをダウンロード
from prophet import Prophet  # 時系列データの予測のためのツール
import plotly.graph_objects as go  # ロットを作成

# Yahoo Financeのデータをダウンロードするための設定
yf.pdr_override()

# Streamlitアプリのタイトル
st.title("株価予測アプリ")

# ユーザーに対話的なウィジェットを提供
# 日本語に変更：株価の銘柄コードを入力
ticker = st.text_input("株価の銘柄コードを入力してください:", "AMZN")
# 日本語に変更：開始日を選択
start_date = st.date_input("開始日を選択してください:", datetime.date(2014, 1, 1))
# 日本語に変更：終了日を選択
end_date = st.date_input("終了日を選択してください:", datetime.date.today())

# Yahoo Financeからデータの取得
data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)

# Prophet用のデータ形式に変換
df = pd.DataFrame({'ds': data.index, 'y': data['Close']})
df = df.fillna(method='ffill')  # 前の値で欠損値を補完


# モデルの作成と学習
model = Prophet(daily_seasonality=True)
model.fit(df)

# 未来の日付を生成
future = model.make_future_dataframe(periods=365)  # 1年分の未来の日付を生成

# 予測
forecast = model.predict(future)

# 過去データをプロット
# Plotlyを使用して、過去のデータのためのキャンドルスティックチャートと予測データのための折れ線グラフを作成
st.plotly_chart(go.Figure(data=[
    go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], 
                   increasing=dict(line=dict(color='#68666c')),  # 陽線の色を変更
                   decreasing=dict(line=dict(color='#d2ddde'))),  # 陰線の色を赤に変更
    go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='予測価格', line=dict(color='#ee7d50'))
], layout=dict(title='ProphetとPlotlyを使用した株価予測', xaxis_title='日付', yaxis_title='株価', template='plotly_dark')))


# データの詳細を表示
st.write("実際の株価データ:")
st.write(data)
st.write("予測された株価データ:")
st.write(forecast)
