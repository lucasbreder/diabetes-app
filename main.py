import pandas as pd
import numpy as np
import joblib
import os
from llm_interpreter import interpretar_resultado

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def carregar_sistema():
    print("Carregando sistema de IA...")
    try:
        modelo = joblib.load('model_diabetes.pkl')
        imputer = joblib.load('imputer.pkl')
        scaler = joblib.load('scaler.pkl')
        return modelo, imputer, scaler
    except FileNotFoundError:
        print("\n⚠️  ALERTA: Modelos não encontrados. Realize o treinamento executando o script train_model.py")
        exit()

def obter_input_usuario():
    print("\n--- FORMULÁRIO DE PACIENTE ---")
    print("Por favor, insira os dados clínicos:")
    
    try:
        # Usamos float() para converter o texto em número
        pregnancies = float(input("1. Número de Gravidezes: "))
        glucose = float(input("2. Glicose (mg/dL): "))
        blood_pressure = float(input("3. Pressão Sanguínea diastólica (mm Hg): "))
        skin_thickness = float(input("4. Espessura da Pele (mm): "))
        insulin = float(input("5. Insulina (mu U/ml): "))
        bmi = float(input("6. Índice de Massa Corporal (IMC): "))
        dpf = float(input("7. Histórico Familiar (Diabetes Pedigree 0.0 - 2.5): "))
        age = float(input("8. Idade: "))
        
        # Criar um DataFrame com UMA linha e as mesmas colunas do treino
        dados = pd.DataFrame([{
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }])
        
        return dados
    except ValueError:
        print("\n[ERRO] Digite apenas números válidos (use ponto para decimais).")
        return None

def pre_processar_input(df, imputer, scaler):
    # 1. Tratamento de Zeros 
    colunas_erro = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[colunas_erro] = df[colunas_erro].replace(0, np.nan)
    
    # 2. Imputação 
    df_imputed_array = imputer.transform(df)
    df_imputed = pd.DataFrame(df_imputed_array, columns=df.columns)
    
    # 3. Feature Engineering 
    df_imputed['Glucose_Insulin'] = df_imputed['Glucose'] * df_imputed['Insulin']
    
    # 4. Escalonamento
    df_final_array = scaler.transform(df_imputed)
    
    # --- CORREÇÃO AQUI ---
    # O scaler devolve um array numpy puro. 
    # Vamos transformá-lo de volta em DataFrame usando as colunas que já sabemos.
    
    df_final = pd.DataFrame(df_final_array, columns=df_imputed.columns)
    
    return df_final

def main():
    limpar_tela()
    print("========================================")
    print("   SISTEMA DE DIAGNÓSTICO DE DIABETES   ")
    print("          Powered by Python AI          ")
    print("========================================")
    
    modelo, imputer, scaler = carregar_sistema()
    
    while True:
        dados_paciente = obter_input_usuario()
        
        if dados_paciente is not None:
            # Preparar os dados
            X_final = pre_processar_input(dados_paciente, imputer, scaler)
            
            # Fazer a previsão
            predicao = modelo.predict(X_final)[0] # [0] pega o primeiro item do array
            probabilidade = modelo.predict_proba(X_final)[0] # Pega as chances %
            
            print("\n" + "="*40)
            print("       RELATÓRIO DO DIAGNÓSTICO")
            print("="*40)
            
            if predicao == 1:
                print(f"🛑 RESULTADO: POSITIVO PARA DIABETES")
                print(f"⚠️  Certeza do Modelo: {probabilidade[1]:.2%}")
                print("\nRecomendação: Procure um médico endocrinologista.")
            else:
                print(f"✅ RESULTADO: NEGATIVO (SAUDÁVEL)")
                print(f"🛡️  Certeza do Modelo: {probabilidade[0]:.2%}")
                print("\nRecomendação: Mantenha hábitos saudáveis.")
            
            # Interpretação via LLM
            prob_final = probabilidade[1] if predicao == 1 else probabilidade[0]
            dados_dict = dados_paciente.iloc[0].to_dict()
            
            print("\n" + "="*40)
            print("   🤖 INTERPRETAÇÃO DA IA (LLama)")
            print("="*40)
            
            interpretacao = interpretar_resultado(dados_dict, predicao, prob_final)
            if interpretacao:
                print(interpretacao)
            else:
                print("(Interpretação indisponível. Verifique se o Ollama está rodando: brew services start ollama)")
            
        print("\n" + "-"*40)
        continuar = input("Diagnosticar outro paciente? (s/n): ").lower()
        if continuar != 's':
            print("Encerrando sistema...")
            break
        limpar_tela()

if __name__ == "__main__":
    main()