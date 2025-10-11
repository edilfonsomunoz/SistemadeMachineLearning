import pandas as pd
import numpy as np

class DataHandler:
    @staticmethod
    def generate_random_data(rows=100, columns=5):
        data = {}
        
        for i in range(columns):
            if i == 0:
                data[f'target'] = np.random.randn(rows)
            else:
                data[f'feature_{i}'] = np.random.randn(rows)
        
        df = pd.DataFrame(data)
        return df
    
    @staticmethod
    def prepare_data(df, target_column, feature_columns):
        if not feature_columns:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns].select_dtypes(include=[np.number])
        y = df[target_column]
        
        X = X.fillna(X.mean())
        
        return X, y, feature_columns
