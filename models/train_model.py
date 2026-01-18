import pandas as pd
from pre_processor.dataset_pre_processor import dataset_pre_processor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib


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
        max_iter=1000  # <--- Adicione isso. O padrão é 100 e costuma ser pouco.
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
        print(f"{model.__class__.__name__}")
        print("-" * 50)
        print(classification_report(y_test, y_pred))
        print("-" * 50)

    # Supondo que 'model' é o seu melhor modelo (ex: LogisticRegression ou RandomForest)
    # E que você tem os objetos 'imputer' e 'scaler' usados no pré-processamento

    print("Salvando o modelo e os processadores...")

    print("Escolhendo e salvando o melhor modelo...")

    # 1. Definir o modelo vencedor (sem a vírgula no final!)
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