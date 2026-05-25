import streamlit as st
from utils.shared_styles import aplicar_estilos_globais

# ========================================
# Configuração da Página
# ========================================
st.set_page_config(
    page_title="Sistema de Saúde da Mulher",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

aplicar_estilos_globais()

# ========================================
# Header
# ========================================
st.markdown("""
<div class="main-header">
    <h1>🏥 Sistema Inteligente de Saúde da Mulher</h1>
    <p>Triagem Ginecológica • Acompanhamento Obstétrico • Detecção de Violência • Prevenção</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# Grade de Navegação (Cards)
# ========================================
st.markdown("### 🗂️ Selecione um Fluxo")
st.markdown("<br>", unsafe_allow_html=True)

# Primeira linha (3 cards)
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("<h3 style='color: #c2185b;'>🩺 Triagem Ginecológica</h3>", unsafe_allow_html=True)
        st.markdown("Análise de sintomas, classificação de urgência, sugestão de exames e agendamento.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.page_link("pages/1_🩺_Triagem_Ginecológica.py", label="Acessar Triagem", icon="➡️")

with col2:
    with st.container(border=True):
        st.markdown("<h3 style='color: #2e7d32;'>🤰 Acompanhamento Obstétrico</h3>", unsafe_allow_html=True)
        st.markdown("Avaliação de risco gestacional, exames por trimestre e alertas de urgência.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.page_link("pages/3_🤰_Obstétrico.py", label="Acessar Obstétrico", icon="➡️")

with col3:
    with st.container(border=True):
        st.markdown("<h3 style='color: #e65100;'>🛡️ Violência Doméstica</h3>", unsafe_allow_html=True)
        st.markdown("Identificação de sinais, protocolo de segurança confidencial e acionamento.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.page_link("pages/2_🛡️_Violência_Doméstica.py", label="Acessar Detecção", icon="➡️")

st.markdown("<br>", unsafe_allow_html=True)

# Segunda linha (2 cards centralizados, usando 3 colunas e deixando a última vazia)
col4, col5, col6 = st.columns(3)

with col4:
    with st.container(border=True):
        st.markdown("<h3 style='color: #1565c0;'>💊 Prevenção</h3>", unsafe_allow_html=True)
        st.markdown("Rastreamento de exames preventivos em atraso, plano de ação e agendamento.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.page_link("pages/4_💊_Prevenção.py", label="Acessar Prevenção", icon="➡️")

with col5:
    with st.container(border=True):
        st.markdown("<h3 style='color: #6b21a8;'>🤖 Assistente Médico Interativo</h3>", unsafe_allow_html=True)
        st.markdown("Módulo com chat clínico baseando-se no prontuário e protocolos FEBRASGO/MS.")
        st.markdown("<br>", unsafe_allow_html=True)
        st.page_link("pages/assistente_medico.py", label="Acessar Assistente", icon="➡️")

st.divider()

st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 2rem;">
    <p>Utilize as abas acima para navegar entre as diferentes funcionalidades de saúde integradas com IA (LangGraph + Ollama).</p>
</div>
""", unsafe_allow_html=True)
