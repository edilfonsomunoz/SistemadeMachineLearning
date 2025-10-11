import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data_handler import DataHandler

class LogisticModels:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
    
    def run_model(self, df, target_column, feature_columns=None):
        X, y_orig, feature_columns = DataHandler.prepare_data(df, target_column, feature_columns)
        
        le = LabelEncoder()
        y = le.fit_transform(y_orig)
        
        if len(np.unique(y)) > 2:
            self.model = LogisticRegression(max_iter=1000, multi_class='ovr')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        avg_method = 'binary' if len(np.unique(y)) == 2 else 'weighted'
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        
        plt.figure(figsize=(12, 5))
        
        if len(np.unique(y)) == 2:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            sigmoid_x = np.linspace(-6, 6, 100)
            sigmoid_y = 1 / (1 + np.exp(-sigmoid_x))
            plt.plot(sigmoid_x, sigmoid_y)
            plt.xlabel('z (log-odds)')
            plt.ylabel('Probabilidad')
            plt.title('Función Sigmoide')
            plt.grid(True)
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            
            plt.subplot(1, 2, 1)
            classes = np.unique(y)
            for i, class_label in enumerate(classes):
                y_test_binary = (y_test == class_label).astype(int)
                fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, i])
                plt.plot(fpr, tpr, label=f'Clase {le.inverse_transform([class_label])[0]}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curvas ROC (Multiclase)')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.bar(range(len(classes)), [np.sum(y_test == c) for c in classes])
            plt.xlabel('Clase')
            plt.ylabel('Frecuencia')
            plt.title('Distribución de Clases')
            plt.xticks(range(len(classes)), le.inverse_transform(classes))
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        coefficients = self.model.coef_.tolist() if hasattr(self.model, 'coef_') else []
        intercept = self.model.intercept_.tolist() if hasattr(self.model, 'intercept_') else []
        
        return {
            'model_type': 'logistic_regression',
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc)
            },
            'coefficients': coefficients,
            'intercept': intercept,
            'feature_names': feature_columns,
            'classes': le.classes_.tolist(),
            'plot': img_base64,
            'predictions_sample': {
                'actual': le.inverse_transform(y_test[:10]).tolist(),
                'predicted': le.inverse_transform(y_pred[:10]).tolist(),
                'probabilities': y_pred_proba[:10].tolist()
            }
        }
