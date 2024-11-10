from flask import Flask, request, jsonify, render_template
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model1')
def model1():
    return render_template('model1.html')

@app.route('/model2')
def model2():
    return render_template('model2.html')

@app.route('/predict_SVM', methods=['POST'])
def predict_SVM():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # 下载股票数据
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)

    # 使用前几天的收盘价及移动平均等特征进行预测
    data['Prev Close'] = data['Close'].shift(1)
    data['5-day MA'] = data['Close'].rolling(window=5).mean()
    data['10-day MA'] = data['Close'].rolling(window=10).mean()
    data['20-day MA'] = data['Close'].rolling(window=20).mean()
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data.dropna(inplace=True)

    # 提取特征和标签
    X = data[['Prev Close', '5-day MA', '10-day MA', '20-day MA', 'Volatility']]
    y = data['Close'].values.ravel()

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # 参数调优
    param_grid = {
        'C': [1, 10, 100, 1000],
        'gamma': [0.001, 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 0.5, 1]
    }
    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # 使用最佳参数创建并训练支持向量回归模型
    model = SVR(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], epsilon=best_params['epsilon'])
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算误差
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 绘制预测结果与实际结果的折线图
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'].iloc[-len(y_test):], y_test, label='Actual price', color='blue')
    plt.plot(data['Date'].iloc[-len(y_test):], y_pred, label='Forecast price', color='red', linestyle='--')
    plt.xlabel('date')
    plt.ylabel('Closing price')
    plt.title(f'{ticker} Stock Price prediction (Support Vector Machine)')
    plt.legend()

    # 将图表保存到内存文件
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'plot_url': f'data:image/png;base64,{plot_url}'
    })

if __name__ == '__main__':
    app.run(debug=True)
