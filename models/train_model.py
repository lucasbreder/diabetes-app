import pandas as pd
from pre_processor.dataset_pre_processor import dataset_pre_processor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib
import numpy as np
import shap


def importance_analysis(model, X_train, X_test, model_name, feature_names):
    """
    Função auxiliar para plotar Feature Importance e SHAP
    """
    print(f"\n--- Analisando Explicalidade: {model_name} ---")
    
    plt.figure(figsize=(10, 6))
    
    # A. Feature Importance (Nativo)
    try:
        if hasattr(model, 'feature_importances_'):
            # Para Random Forest / Decision Tree
            importances = model.feature_importances_
            indices = np.argsort(importances)
            plt.title(f'Feature Importance - {model_name}')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Importância Relativa')
            print("--- Gerando gráfico...")
            plt.savefig(f'./graphs/feature_importance_{model_name}.png')
            print("--- Gráfico salvo como feature_importance.png")
            
        elif hasattr(model, 'coef_'):
            # Para Regressão Logística (Coeficientes)
            # Pegamos o valor absoluto para ver a força, e a cor indica a direção
            coefs = model.coef_[0]
            indices = np.argsort(abs(coefs))
            plt.title(f'Coeficientes (Pesos) - {model_name}')
            plt.barh(range(len(indices)), coefs[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Peso (Negativo = Proteção / Positivo = Risco)')
            print("--- Gerando gráfico...")
            plt.savefig(f'./graphs/feature_importance_{model_name}.png')
            print("--- Gráfico salvo como feature_importance.png")
            
    except Exception as e:
        print(f"Não foi possível gerar gráfico de importância simples: {e}")


def train_model():
    # 1. Carregar e Pré-processar
    df = pd.read_csv("./dataset/diabetes.csv")
    df_normalized, imputer, scaler = dataset_pre_processor(df)
    X = df_normalized.drop("Outcome", axis=1) 
    y = df_normalized["Outcome"]

    # 2. Divisão Treino/Teste mantendo a proporção de classes
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Modelos

    models = [
        KNeighborsClassifier(n_neighbors=5),
        DecisionTreeClassifier(),
        LogisticRegression(
        class_weight="balanced",
        max_iter=1000 
        ),
        RandomForestClassifier(
        n_estimators=500,        
        min_samples_leaf=1,     
        class_weight="balanced_subsample",
        random_state=42
    )
    ]

    # 4. Resultados

    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        importance_analysis(model, X_train, X_test, model.__class__.__name__, X_train.columns)
        print(f"{model.__class__.__name__}")
        print("-" * 50)
        print(classification_report(y_test, y_pred))
        print("-" * 50)

    print("Salvando o modelo e os processadores...")

    print("Salvando modelo...")

    # 1. Definir o modelo vencedor
    final_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )

    # 2. TREINAR o modelo final (CRUCIAL: Você precisa ensinar ele antes de salvar)
    final_model.fit(X_train, y_train)

    # 3. Salvar
    # Note que estou salvando 'final_model', que acabamos de treinar
    joblib.dump(final_model, 'model_diabetes.pkl')

    # 4. Salvar os processadores (Esses já vieram treinados da função dataset_pre_processor)
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Arquivos salvos com sucesso: .pkl")

train_model()