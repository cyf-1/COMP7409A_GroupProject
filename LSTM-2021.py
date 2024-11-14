#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[2]:


# 下载日度数据
ticker = 'AAPL'  # 可以替换为其他股票代码
data = yf.download(ticker, start="2021-01-01", end="2024-01-01")
print(f"Dataset size: {data.shape[0]} samples")

# 仅保留收盘价
data = data[['Close']]


# In[5]:


# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data[['Close']])

# 定义创建滑动窗口数据集的函数
def create_windowed_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# 使用滑动窗口创建数据集，假设使用前 10 天的数据预测第 11 天
window_size = 10
X, y = create_windowed_dataset(data['Close'].values, window_size)


# In[7]:


# 划分数据集（80% 训练集，20% 测试集）
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 将数据 reshape 为 LSTM 所需的 3D 形状：[样本数, 时间步, 特征数]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[9]:


# 构建 LSTM 模型
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)


# In[11]:


# 预测测试集
y_pred = model.predict(X_test)

# 逆归一化
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_rescaled = scaler.inverse_transform(y_pred)

# 计算均方误差
mse = np.mean((y_test_rescaled - y_pred_rescaled) ** 2)
print(f"Mean Squared Error: {mse:.2f}")

# 计算均方根误差 RMSE
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# 计算平均绝对误差 MAE
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 计算平均绝对百分比误差 MAPE
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# In[13]:


# 将实际值和预测值放入 DataFrame 中以便于可视化
test_dates = data.index[-len(y_test):]
results = pd.DataFrame({'Actual': y_test_rescaled.flatten(), 'Predicted': y_pred_rescaled.flatten()}, index=test_dates)

# 绘制图表
plt.figure(figsize=(14, 7))
plt.plot(results.index, results['Actual'], label='Actual Price', color='blue')
plt.plot(results.index, results['Predicted'], label='Predicted Price', color='red')

plt.title('Daily Stock Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




