import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# 数据
depth_camera = np.array([])
depth_groundtruth = np.array([])


def simplified_error_model(d, a, b, f, g):
    return (1 + a) * d + b * d**2 + f + g / d

# 在完整数据上拟合模型
X_full = depth_camera.reshape(-1, 1)
params, _ = curve_fit(simplified_error_model, X_full.flatten(), depth_groundtruth)
y_full = depth_groundtruth - simplified_error_model(X_full.flatten(), *params)

X_full_extended = np.column_stack([X_full, X_full**2, X_full**3, np.log(X_full + 1), 1/X_full])
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full_extended)

best_model = RandomForestRegressor(max_depth=10, n_estimators=200, random_state=42)
best_model.fit(X_full_scaled, y_full)

y_full_pred = simplified_error_model(depth_camera, *params) + best_model.predict(X_full_scaled)

rmse_full = np.sqrt(np.mean((depth_groundtruth - y_full_pred)**2))
print(f"RMSE on full data (simplified model): {rmse_full:.2f} mm")

joblib.dump(scaler, 'scaler_simplified.pkl')
joblib.dump(best_model, 'rf_model_simplified.pkl')
np.save('params_simplified.npy', params)

print("Simplified model saved successfully!")