import pandas as pd
import numpy as np
import joblib
import sys
import os

# Importando a função do LLM da raiz
from llm_interpreter import interpretar_resultado

def rodar_demonstracao():
    print("="*50)
    print("🩺 DEMONSTRAÇÃO DO PIPELINE DE IA (DIABETES)")
    print("="*50)

    # 1. Carregar modelos
    print("\n[1/4] Carregando modelos...")
    try:
        modelo = joblib.load('model_diabetes.pkl')
        imputer = joblib.load('imputer.pkl')
        scaler = joblib.load('scaler.pkl')
        print("✅ Modelos carregados com sucesso!")
    except FileNotFoundError:
        print("❌ ERRO: Modelos não encontrados. Rode primeiro: `python models/train_model.py`")
        sys.exit(1)

    # 2. Dados de exemplo
    dados_paciente = {
        'Pregnancies': 4,
        'Glucose': 130,
        'BloodPressure': 80,
        'SkinThickness': 22,
        'Insulin': 95,
        'BMI': 29.5,
        'DiabetesPedigreeFunction': 0.75,
        'Age': 48
    }
    df_paciente = pd.DataFrame([dados_paciente])
    
    print("\n[2/4] Dados mockados do paciente inseridos.")
    for k, v in dados_paciente.items():
        print(f"  - {k}: {v}")

    # 3. Processamento e Predição
    print("\n[3/4] Processando e executando predição de Machine Learning...")
    colunas_erro = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_paciente[colunas_erro] = df_paciente[colunas_erro].replace(0, np.nan)
    
    df_imputed = pd.DataFrame(imputer.transform(df_paciente), columns=df_paciente.columns)
    df_imputed['Glucose_Insulin'] = df_imputed['Glucose'] * df_imputed['Insulin']
    
    X_final = pd.DataFrame(scaler.transform(df_imputed), columns=df_imputed.columns)
    
    predicao = modelo.predict(X_final)[0]
    probabilidades = modelo.predict_proba(X_final)[0]
    
    resultado_str = "POSITIVO" if predicao == 1 else "NEGATIVO"
    confianca = probabilidades[1] if predicao == 1 else probabilidades[0]
    
    print(f"  🤖 Resultado do ML: {resultado_str} (Confiança: {confianca:.2%})")

    # 4. LLM
    print("\n[4/4] Consultando a LLM para análise textual (LLama via Ollama)...")
    texto_llm = interpretar_resultado(dados_paciente, predicao, confianca)
    
    print("\n" + "="*50)
    print("📝 LAUDO GERADO PELA IA:")
    print("="*50)
    if texto_llm:
        print(texto_llm)
    else:
        print("⚠️ Não foi possível conectar ao Ollama. Verifique se ele está rodando.")

if __name__ == "__main__":
    rodar_demonstracao()
