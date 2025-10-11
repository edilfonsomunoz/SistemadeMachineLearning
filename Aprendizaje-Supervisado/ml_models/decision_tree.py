import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data_handler import DataHandler

class DecisionTreeModels:
    def __init__(self):
        self.model = None
    
    def run_model(self, df, model_type, target_column, feature_columns=None, max_depth=None):
        X, y_orig, feature_columns = DataHandler.prepare_data(df, target_column, feature_columns)
        
        is_classification = df[target_column].dtype == 'object' or len(df[target_column].unique()) < 20
        
        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(y_orig)
            
            if model_type == 'id3':
                self.model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
            elif model_type == 'c45':
                self.model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
            else:
                self.model = DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
        else:
            y = y_orig
            le = None
            self.model = DecisionTreeRegressor(max_depth=max_depth)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        feature_importance = dict(zip(feature_columns, self.model.feature_importances_.tolist()))
        
        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {
                'accuracy': float(accuracy),
                'tree_depth': int(self.model.get_depth()),
                'n_leaves': int(self.model.get_n_leaves()),
                'n_features': len(feature_columns)
            }
            
            predictions_sample = {
                'actual': le.inverse_transform(y_test[:10]).tolist(),
                'predicted': le.inverse_transform(y_pred[:10].astype(int)).tolist()
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            metrics = {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'tree_depth': int(self.model.get_depth()),
                'n_leaves': int(self.model.get_n_leaves()),
                'n_features': len(feature_columns)
            }
            
            predictions_sample = {
                'actual': y_test[:10].tolist(),
                'predicted': y_pred[:10].tolist()
            }
        
        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            feature_names=feature_columns,
            class_names=le.classes_.tolist() if is_classification else None,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title(f'Árbol de Decisión - {model_type.upper()}')
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False)
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importancia')
        plt.ylabel('Variable')
        plt.title('Importancia de Variables')
        plt.tight_layout()
        
        importance_buffer = BytesIO()
        plt.savefig(importance_buffer, format='png', dpi=100, bbox_inches='tight')
        importance_buffer.seek(0)
        importance_base64 = base64.b64encode(importance_buffer.read()).decode()
        plt.close()
        
        return {
            'model_type': f'decision_tree_{model_type}',
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_names': feature_columns,
            'plot': img_base64,
            'importance_plot': importance_base64,
            'predictions_sample': predictions_sample,
            'algorithm_info': {
                'id3': 'Utiliza ganancia de información (entropía)',
                'c45': 'Mejora de ID3 con manejo de valores continuos',
                'cart': 'Utiliza índice de Gini para clasificación'
            }.get(model_type, '')
        }
