# ==========================================
# 📌 IMPORTS
# ==========================================

# Pandas para manipular DataFrame nos testes
import pandas as pd

# Classes usadas dentro da função (para validar tipo de retorno)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

# Função que vamos testar
from pre_processor.dataset_pre_processor import dataset_pre_processor


# ==========================================
# 📌 FUNÇÃO AUXILIAR PARA CRIAR DADOS DE TESTE
# ==========================================

def create_sample_dataset(with_outcome=True):
    """
    Cria um DataFrame de exemplo com dados semelhantes ao dataset de diabetes.

    Parâmetro:
    - with_outcome: define se a coluna 'Outcome' será incluída

    Retorna:
    - DataFrame pronto para ser usado nos testes
    """

    data = {
        "Pregnancies": [2, 4, 1, 0, 3, 5, 2, 1, 6, 2, 3],
        "Glucose": [120, 0, 98, 140, 110, 130, 115, 100, 150, 125, 117],
        "BloodPressure": [70, 80, 0, 90, 75, 85, 72, 68, 88, 77, 79],
        "SkinThickness": [20, 35, 25, 0, 30, 28, 26, 24, 32, 29, 27],
        "Insulin": [85, 0, 88, 96, 110, 105, 98, 90, 120, 100, 95],
        "BMI": [28.5, 33.6, 0, 35.1, 29.4, 31.2, 30.0, 27.8, 36.5, 32.1, 29.9],
        "DiabetesPedigreeFunction": [0.4, 0.6, 0.2, 0.8, 0.5, 0.7, 0.3, 0.25, 0.9, 0.45, 0.55],
        "Age": [35, 50, 28, 45, 31, 52, 39, 26, 60, 41, 33],
    }

    df = pd.DataFrame(data)

    # Se quiser testar com variável target
    if with_outcome:
        df["Outcome"] = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0]

    return df


# ==========================================
# 📌 TESTE 1 - RETORNO COM OUTCOME
# ==========================================

def test_returns_processed_dataset_with_outcome():
    """
    Testa se a função retorna corretamente quando o dataset possui a coluna 'Outcome'.

    Esperamos:
    - um DataFrame processado
    - um objeto KNNImputer
    - um objeto RobustScaler
    """

    df = create_sample_dataset(with_outcome=True)

    processed_df, imputer, scaler = dataset_pre_processor(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)
    assert isinstance(imputer, KNNImputer)
    assert isinstance(scaler, RobustScaler)


# ==========================================
# 📌 TESTE 2 - RETORNO SEM OUTCOME
# ==========================================

def test_returns_processed_dataset_without_outcome():
    """
    Testa se a função funciona corretamente quando NÃO existe a coluna 'Outcome'.

    Esperamos:
    - um DataFrame processado
    - um imputer
    - um scaler
    """

    df = create_sample_dataset(with_outcome=False)

    processed_df, imputer, scaler = dataset_pre_processor(df)

    assert processed_df is not None
    assert isinstance(processed_df, pd.DataFrame)
    assert isinstance(imputer, KNNImputer)
    assert isinstance(scaler, RobustScaler)


# ==========================================
# 📌 TESTE 3 - FEATURE ENGINEERING
# ==========================================

def test_creates_glucose_insulin_column():
    """
    Verifica se a feature 'Glucose_Insulin' foi criada.

    Essa é uma regra de negócio importante do modelo.
    """

    df = create_sample_dataset(with_outcome=False)

    processed_df, _, _ = dataset_pre_processor(df)

    assert "Glucose_Insulin" in processed_df.columns


# ==========================================
# 📌 TESTE 4 - PRESERVAÇÃO DO TARGET
# ==========================================

def test_keeps_outcome_column_when_present():
    """
    Verifica se a coluna 'Outcome' continua presente após o processamento.
    """

    df = create_sample_dataset(with_outcome=True)

    processed_df, _, _ = dataset_pre_processor(df)

    assert "Outcome" in processed_df.columns


# ==========================================
# 📌 TESTE 5 - NÃO DEVE HAVER NaN APÓS IMPUTAÇÃO
# ==========================================

def test_does_not_keep_nan_after_processing():
    """
    Após a imputação com KNNImputer, não deve haver valores nulos.
    """

    df = create_sample_dataset(with_outcome=False)

    processed_df, _, _ = dataset_pre_processor(df)

    assert processed_df.isnull().sum().sum() == 0


# ==========================================
# 📌 TESTE 6 - QUANTIDADE DE LINHAS
# ==========================================

def test_returns_same_number_of_rows():
    """
    O pré-processamento não deve remover nem duplicar linhas.
    """

    df = create_sample_dataset(with_outcome=True)

    processed_df, _, _ = dataset_pre_processor(df)

    assert len(processed_df) == len(df)