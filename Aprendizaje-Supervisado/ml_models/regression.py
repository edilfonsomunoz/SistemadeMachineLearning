import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data_handler import DataHandler

class RegressionModels:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'polynomial': None,
            'rbf': KernelRidge(kernel='rbf', alpha=1.0),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0)
        }
    
    def run_model(self, df, model_type, target_column, feature_columns=None):
        X, y, feature_columns = DataHandler.prepare_data(df, target_column, feature_columns)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == 'polynomial':
            poly = PolynomialFeatures(degree=2)
            X_train = poly.fit_transform(X_train)
            X_test = poly.transform(X_test)
            model = LinearRegression()
        else:
            model = self.models.get(model_type, LinearRegression())
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        if hasattr(model, 'coef_'):
            coefficients = model.coef_.tolist() if isinstance(model.coef_, np.ndarray) else [model.coef_]
        else:
            coefficients = []
        
        intercept = float(model.intercept_) if hasattr(model, 'intercept_') else None
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title(f'Predicciones vs Valores Reales - {model_type}')
        
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred_test
        plt.scatter(y_pred_test, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicciones')
        plt.ylabel('Residuos')
        plt.title('Análisis de Residuos')
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return {
            'model_type': model_type,
            'metrics': {
                'mse_train': float(mse_train),
                'mse_test': float(mse_test),
                'r2_train': float(r2_train),
                'r2_test': float(r2_test),
                'mae_test': float(mae_test),
                'rmse_test': float(np.sqrt(mse_test))
            },
            'coefficients': coefficients,
            'intercept': intercept,
            'feature_names': feature_columns,
            'plot': img_base64,
            'predictions_sample': {
                'actual': y_test.head(10).tolist(),
                'predicted': y_pred_test[:10].tolist()
            }
        }
