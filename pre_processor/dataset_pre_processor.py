from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np

def dataset_pre_processor(dataset):
    # 1. Separar X e y (Guardando o índice original)
    if "Outcome" in dataset.columns:
        X = dataset.drop('Outcome', axis=1)
        y = dataset['Outcome']
    else:
        X = dataset
        y = None
    
    # 2. Substituir zeros por NaN
    colunas_erro = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    X[colunas_erro] = X[colunas_erro].replace(0, np.nan)

    # 3. IMPUTAÇÃO
    imputer = KNNImputer(n_neighbors=10)
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=dataset.index)

    # 4. FEATURE ENGINEERING
    X['Glucose_Insulin'] = X['Glucose'] * X['Insulin']
    
    # 5. ESCALONAMENTO
    scaler_final = RobustScaler()
    X_final_array = scaler_final.fit_transform(X)
    X_final = pd.DataFrame(X_final_array, columns=X.columns, index=dataset.index)

    # 6. RETORNO 
    if y is not None:
        dataset_completo = pd.concat([X_final, y], axis=1)
        return dataset_completo, imputer, scaler_final
    else:
        return X_final, imputer, scaler_final