
import pandas as pd; from sklearn.linear_model import LinearRegression
data = pd.read_pickle('2025C-sub1-preprocessed.pkl')
month_cols = [c for c in data.columns if c.startswith('m_202')]
X = data[['gw_c','bmi_c','pc1'] + month_cols].values; y = data['y_log'].values
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
R2, RMSE = sklearn.metrics.r2_score(y, y_pred), np.sqrt(np.mean((y-y_pred)**2))
