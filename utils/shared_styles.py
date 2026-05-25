"""
Estilos e componentes CSS compartilhados entre todas as páginas Streamlit.
"""

import streamlit as st


def aplicar_estilos_globais():
    """Aplica CSS global compartilhado entre todas as páginas."""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* ===== Ajuste do espaçamento nativo do Streamlit ===== */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 1rem !important;
    }
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 3.5rem !important;
    }

    /* ===== Header ===== */
    .main-header {
        text-align: center;
        padding: 0rem 0 1rem 0;
        margin-bottom: 0.5rem;
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

    /* ===== Cards de resultado ===== */
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

    /* ===== Cards temáticos dos fluxos ===== */
    .flow-card {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        color: #1e293b;
    }
    .flow-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    .flow-card h3 { margin: 0 0 0.5rem 0; font-size: 1.1rem; }
    .flow-card p { margin: 0; font-size: 0.9rem; opacity: 0.85; }

    .flow-gineco { background: linear-gradient(135deg, #fce4ec, #f8bbd0); border: 1px solid #f48fb1; }
    .flow-gineco h3 { color: #c2185b; }
    .flow-violencia { background: linear-gradient(135deg, #fff3e0, #ffe0b2); border: 1px solid #ffb74d; }
    .flow-violencia h3 { color: #e65100; }
    .flow-obstetrico { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border: 1px solid #81c784; }
    .flow-obstetrico h3 { color: #2e7d32; }
    .flow-prevencao { background: linear-gradient(135deg, #e3f2fd, #bbdefb); border: 1px solid #64b5f6; }
    .flow-prevencao h3 { color: #1565c0; }

    /* ===== Cards de urgência ===== */
    .urgencia-vermelho {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border: 2px solid #ef4444;
        border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(239,68,68,0.15);
        color: #7f1d1d;
    }
    .urgencia-vermelho h3 { color: #dc2626; }
    .urgencia-amarelo {
        background: linear-gradient(135deg, #fefce8, #fef08a);
        border: 2px solid #eab308;
        border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(234,179,8,0.15);
        color: #713f12;
    }
    .urgencia-amarelo h3 { color: #a16207; }
    .urgencia-verde {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 2px solid #22c55e;
        border-radius: 16px; padding: 1.5rem; margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(34,197,94,0.15);
        color: #14532d;
    }
    .urgencia-verde h3 { color: #16a34a; }

    /* ===== Card LLM ===== */
    .llm-card {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 1px solid #7dd3fc;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #0c4a6e;
    }
    .llm-card h3 { color: #0369a1; }

    /* ===== Card de info/resultado ===== */
    .info-card {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        color: #1e293b;
    }

    /* ===== Alerta de risco ===== */
    .alerta-critico {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        color: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        animation: pulse-alerta 2s infinite;
    }
    @keyframes pulse-alerta {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
        50% { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
    }

    /* ===== Disclaimer ===== */
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

    /* ===== Sidebar ===== */
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

    /* ===== Exame badge ===== */
    .exame-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-urgente { background: #fee2e2; color: #dc2626; }
    .badge-rotina { background: #dbeafe; color: #1d4ed8; }
</style>
""", unsafe_allow_html=True)


def exibir_disclaimer():
    """Exibe o disclaimer médico padrão."""
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Aviso:</strong> Esta análise é gerada por Inteligência Artificial e
        <strong>NÃO substitui</strong> uma consulta médica profissional.
        Procure sempre um médico para avaliação e diagnóstico adequados.
    </div>
    """, unsafe_allow_html=True)


def card_urgencia(codigo: str, titulo: str, conteudo: str):
    """Renderiza um card colorido conforme o nível de urgência."""
    classe = {
        "VERMELHO": "urgencia-vermelho",
        "AMARELO": "urgencia-amarelo",
        "VERDE": "urgencia-verde",
    }.get(codigo, "info-card")
    st.markdown(f"""
    <div class="{classe}">
        <h3>{titulo}</h3>
        <p>{conteudo}</p>
    </div>
    """, unsafe_allow_html=True)
