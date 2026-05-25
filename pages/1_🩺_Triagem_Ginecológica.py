import streamlit as st
from utils.shared_styles import aplicar_estilos_globais, card_urgencia, exibir_disclaimer
from utils.paciente_selector import render_seletor_sidebar
from utils.fluxo_runner import executar_fluxo_com_progresso

st.set_page_config(page_title="Triagem Ginecológica", page_icon="🩺", layout="wide")
aplicar_estilos_globais()

st.markdown("""
<div class="main-header">
    <h1>🩺 Triagem Ginecológica</h1>
    <p>Análise de sintomas • Classificação de urgência • Sugestão de exames • Agendamento</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# Seletor de paciente (sidebar)
# ========================================
contexto = render_seletor_sidebar()

prefill_nome = contexto.nome if contexto else ""
prefill_idade = contexto.idade if contexto else 30
prefill_gestacoes = contexto.gestacoes if contexto else 0
prefill_contraceptivo = contexto.metodo_contraceptivo if contexto else "Nenhum"
prefill_queixas = contexto.queixas if contexto else ""

OPCOES_CONTRACEPTIVO = [
    "Nenhum", "Pílula combinada", "Pílula progestágeno", "DIU de cobre",
    "DIU hormonal (Mirena)", "Implante", "Injetável", "Preservativo", "Outro",
]


def _encontrar_indice(lista: list[str], valor: str | None, fallback: int = 0) -> int:
    """Localiza o índice do valor na lista, tolerando substrings (ex.: 'DIU hormonal' ∈ 'DIU hormonal (Mirena)')."""
    if not valor:
        return fallback
    valor_low = valor.lower()
    for i, op in enumerate(lista):
        if op.lower() == valor_low or op.lower() in valor_low or valor_low in op.lower():
            return i
    return fallback


# ========================================
# Formulário
# ========================================
with st.form("form_triagem_gineco"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 Dados da Paciente")
        nome = st.text_input("Nome da paciente", value=prefill_nome, placeholder="Ex: Maria Silva")
        idade = st.number_input("Idade", min_value=10, max_value=100, value=prefill_idade)
        gestacoes = st.number_input("Gestações anteriores", min_value=0, max_value=20, value=prefill_gestacoes)
        contraceptivo = st.selectbox(
            "Uso de contraceptivo",
            OPCOES_CONTRACEPTIVO,
            index=_encontrar_indice(OPCOES_CONTRACEPTIVO, prefill_contraceptivo),
        )

    with col2:
        st.markdown("#### 📋 Histórico")
        historico_menstrual = st.text_area(
            "Histórico menstrual",
            placeholder="Ex: Ciclos regulares de 28 dias, fluxo moderado",
            value=(
                f"Última menstruação: {contexto.ultima_menstruacao}"
                if contexto and contexto.ultima_menstruacao else ""
            ),
            height=80,
        )
        ultima_consulta = st.text_input("Última consulta ginecológica", placeholder="Ex: 2024-06-15")
        hist_familiar = st.multiselect("Histórico familiar", [
            "Câncer de mama", "Câncer de colo uterino", "Câncer de ovário",
            "Mioma uterino", "Endometriose", "SOP", "Outro"
        ])

    st.markdown("#### 🔍 Queixas Atuais")
    sintomas = st.multiselect("Sintomas relatados", [
        "Dor pélvica", "Corrimento anormal", "Sangramento intenso",
        "Irregularidade menstrual", "Dispareunia (dor na relação)",
        "Prurido genital", "Dor abdominal aguda", "Sangramento pós-menopausa",
        "Massa palpável", "Febre alta", "Amenorreia", "Dismenorreia intensa",
    ])
    queixas_extra = st.text_area(
        "Queixas adicionais",
        value=prefill_queixas,
        placeholder="Descreva outros sintomas...",
        height=60,
    )

    submitted = st.form_submit_button("🚀 Executar Triagem", use_container_width=True, type="primary")

# ========================================
# Execução do fluxo
# ========================================
if submitted:
    if not nome or not sintomas:
        st.warning("⚠️ Preencha o nome da paciente e selecione ao menos um sintoma.")
        st.stop()

    from flows.triagem_ginecologica import criar_fluxo_triagem_ginecologica

    paciente_id_str = (
        f"PAC-{contexto.paciente_id:04d}" if contexto else f"PAC-{hash(nome) % 10000:04d}"
    )

    entrada = {
        "paciente_id": paciente_id_str,
        "nome_paciente": nome,
        "idade": idade,
        "sintomas": sintomas,
        "historico_menstrual": historico_menstrual or "Não informado",
        "uso_contraceptivo": contraceptivo,
        "gestacoes_anteriores": gestacoes,
        "ultima_consulta_gineco": ultima_consulta or "Não informada",
        "historico_familiar": hist_familiar,
        "queixas_adicionais": queixas_extra or "Nenhuma",
    }

    fluxo = criar_fluxo_triagem_ginecologica()
    resultado = executar_fluxo_com_progresso(
        fluxo,
        entrada,
        titulo="⏳ Executando triagem ginecológica...",
        titulo_final="✅ Triagem concluída",
        rotulos_nos={
            "analisar_sintomas": "Analisando sintomas e fatores de risco",
            "classificar_urgencia": "Classificando urgência (vermelho/amarelo/verde)",
            "sugerir_exames": "Sugerindo exames complementares",
            "gerar_orientacoes": "Gerando orientações personalizadas (LLM)",
            "realizar_agendamento": "Definindo agendamento apropriado",
        },
    )

    # ========================================
    # Resultados
    # ========================================
    st.divider()
    st.markdown("## 📊 Resultado da Triagem")

    # Classificação de urgência
    urgencia = resultado.get("classificacao_urgencia", {})
    codigo = urgencia.get("codigo", "VERDE")

    emoji_map = {"VERMELHO": "🔴", "AMARELO": "🟡", "VERDE": "🟢"}
    emoji = emoji_map.get(codigo, "⚪")

    card_urgencia(
        codigo,
        f"{emoji} Classificação: {codigo}",
        f"<strong>{urgencia.get('descricao', '')}</strong><br>"
        f"Tempo máximo: {urgencia.get('tempo_maximo', '')}<br>"
        f"Encaminhamento: {urgencia.get('encaminhamento', '')}",
    )

    # Exames sugeridos
    col_ex, col_ag = st.columns(2)

    with col_ex:
        st.markdown("### 🧪 Exames Sugeridos")
        exames = resultado.get("exames_sugeridos", [])
        for e in exames:
            badge = "badge-urgente" if e.get("prioridade") == "urgente" else "badge-rotina"
            st.markdown(
                f'<div class="info-card">'
                f'<strong>{e["nome"]}</strong> '
                f'<span class="exame-badge {badge}">{e.get("prioridade", "rotina").upper()}</span>'
                f'<br><small>{e.get("justificativa", "")}</small></div>',
                unsafe_allow_html=True,
            )

    with col_ag:
        st.markdown("### 📅 Agendamento")
        agendamento = resultado.get("agendamento", {})
        st.markdown(f"""
        <div class="info-card">
            <strong>📅 Data sugerida:</strong> {agendamento.get("data_sugerida", "N/A")}<br>
            <strong>👩‍⚕️ Especialidade:</strong> {agendamento.get("especialidade", "")}<br>
            <strong>📋 Tipo:</strong> {agendamento.get("tipo_consulta", "")}<br>
            <strong>📝 Obs:</strong> {agendamento.get("observacoes", "")}
        </div>
        """, unsafe_allow_html=True)

    # Orientações da IA
    orientacoes = resultado.get("orientacoes", "")
    if orientacoes:
        st.markdown("### 🤖 Orientações da IA")
        st.markdown(f'<div class="llm-card">{orientacoes}</div>', unsafe_allow_html=True)

    exibir_disclaimer()
