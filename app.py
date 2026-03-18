import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from llm_interpreter import interpretar_resultado

# ========================================
# Configuração da Página
# ========================================
st.set_page_config(
    page_title="Diagnóstico de Diabetes",
    page_icon="🩺",
    layout="centered"
)

# ========================================
# CSS Customizado
# ========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1rem;
    }
    
    .result-card {
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }
    .result-positive {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #f87171;
    }
    .result-positive h2 { color: #dc2626; }
    .result-positive p { color: #991b1b; }
    
    .result-negative {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 2px solid #4ade80;
    }
    .result-negative h2 { color: #16a34a; }
    .result-negative p { color: #166534; }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .llm-card {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 1px solid #7dd3fc;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .llm-card h3 { color: #0369a1; }
    
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1.5rem;
        font-size: 0.85rem;
        color: #92400e;
        text-align: center;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #475569;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# Carregar Modelos
# ========================================
@st.cache_resource
def carregar_sistema():
    try:
        modelo = joblib.load('model_diabetes.pkl')
        imputer = joblib.load('imputer.pkl')
        scaler = joblib.load('scaler.pkl')
        return modelo, imputer, scaler
    except FileNotFoundError:
        return None, None, None

def pre_processar_input(df, imputer, scaler):
    colunas_erro = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[colunas_erro] = df[colunas_erro].replace(0, np.nan)
    
    df_imputed_array = imputer.transform(df)
    df_imputed = pd.DataFrame(df_imputed_array, columns=df.columns)
    df_imputed['Glucose_Insulin'] = df_imputed['Glucose'] * df_imputed['Insulin']
    
    df_final_array = scaler.transform(df_imputed)
    df_final = pd.DataFrame(df_final_array, columns=df_imputed.columns)
    return df_final

# ========================================
# Header
# ========================================
st.markdown("""
<div class="main-header">
    <h1>🩺 Sistema de Diagnóstico de Diabetes</h1>
    <p>Powered by Machine Learning + LLama AI</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# Carregar modelo
# ========================================
modelo, imputer, scaler = carregar_sistema()

if modelo is None:
    st.error("⚠️ Modelos não encontrados. Execute primeiro: `python -m models.train_model`")
    st.stop()

# ========================================
# Sidebar - Formulário
# ========================================
with st.sidebar:
    st.markdown("### 📋 Dados do Paciente")
    st.markdown('<div class="sidebar-info">Preencha os dados clínicos abaixo para realizar o diagnóstico.</div>', unsafe_allow_html=True)
    
    with st.form("paciente_form"):
        pregnancies = st.number_input("🤰 Gravidezes", min_value=0, max_value=20, value=None, step=1, placeholder="Ex: 2")
        glucose = st.number_input("🩸 Glicose (mg/dL)", min_value=0.0, max_value=300.0, value=None, step=1.0, placeholder="Ex: 100")
        blood_pressure = st.number_input("💓 Pressão Sanguínea (mm Hg)", min_value=0.0, max_value=200.0, value=None, step=1.0, placeholder="Ex: 70")
        skin_thickness = st.number_input("📏 Espessura da Pele (mm)", min_value=0.0, max_value=100.0, value=None, step=1.0, placeholder="Ex: 20")
        insulin = st.number_input("💉 Insulina (mu U/ml)", min_value=0.0, max_value=900.0, value=None, step=1.0, placeholder="Ex: 80")
        bmi = st.number_input("⚖️ IMC", min_value=0.0, max_value=70.0, value=None, step=0.1, placeholder="Ex: 25.0")
        dpf = st.number_input("🧬 Histórico Familiar (Pedigree)", min_value=0.0, max_value=2.5, value=None, step=0.01, placeholder="Ex: 0.5")
        age = st.number_input("🎂 Idade", min_value=1, max_value=120, value=None, step=1, placeholder="Ex: 30")
        
        submitted = st.form_submit_button("🔍 Realizar Diagnóstico", use_container_width=True)

# ========================================
# Diagnóstico
# ========================================
if submitted:
    campos = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    if any(v is None for v in campos):
        st.warning("⚠️ Preencha todos os campos antes de realizar o diagnóstico.")
        st.stop()
    
    dados_paciente = pd.DataFrame([{
        'Pregnancies': float(pregnancies),
        'Glucose': float(glucose),
        'BloodPressure': float(blood_pressure),
        'SkinThickness': float(skin_thickness),
        'Insulin': float(insulin),
        'BMI': float(bmi),
        'DiabetesPedigreeFunction': float(dpf),
        'Age': float(age)
    }])
    
    X_final = pre_processar_input(dados_paciente, imputer, scaler)
    
    predicao = modelo.predict(X_final)[0]
    probabilidade = modelo.predict_proba(X_final)[0]
    
    # --- Resultado ---
    if predicao == 1:
        certeza = probabilidade[1]
        st.markdown(f"""
        <div class="result-card result-positive">
            <h2>🛑 POSITIVO PARA DIABETES</h2>
            <div class="metric-value">{certeza:.1%}</div>
            <p>Certeza do Modelo</p>
            <p><strong>Recomendação:</strong> Procure um médico endocrinologista.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        certeza = probabilidade[0]
        st.markdown(f"""
        <div class="result-card result-negative">
            <h2>✅ NEGATIVO — SAUDÁVEL</h2>
            <div class="metric-value">{certeza:.1%}</div>
            <p>Certeza do Modelo</p>
            <p><strong>Recomendação:</strong> Mantenha hábitos saudáveis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Barra de confiança ---
    st.markdown("**Nível de Confiança do Modelo:**")
    st.progress(float(certeza))
    
    # --- Interpretação LLM ---
    st.markdown('<div class="llm-card"><h3>🤖 Interpretação da IA (LLama)</h3>', unsafe_allow_html=True)
    
    dados_dict = dados_paciente.iloc[0].to_dict()
    prob_final = probabilidade[1] if predicao == 1 else probabilidade[0]
    
    with st.spinner("Gerando interpretação com LLama..."):
        interpretacao = interpretar_resultado(dados_dict, predicao, prob_final)
    
    if interpretacao:
        st.markdown(interpretacao)
    else:
        st.warning("Interpretação indisponível. Verifique se o Ollama está rodando: `brew services start ollama`")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Disclaimer ---
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Aviso:</strong> Esta análise é gerada por Inteligência Artificial e <strong>NÃO substitui</strong> uma consulta médica profissional.
        Procure sempre um médico para avaliação e diagnóstico adequados.
    </div>
    """, unsafe_allow_html=True)

else:
    # Estado inicial
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #6b7280;">
        <p style="font-size: 3rem; margin-bottom: 1rem;">👈</p>
        <h3>Preencha o formulário na barra lateral</h3>
        <p>Insira os dados clínicos do paciente e clique em <strong>"Realizar Diagnóstico"</strong></p>
    </div>
    """, unsafe_allow_html=True)
