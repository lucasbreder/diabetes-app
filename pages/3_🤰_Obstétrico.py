import streamlit as st
from utils.shared_styles import aplicar_estilos_globais, exibir_disclaimer
from utils.paciente_selector import render_seletor_sidebar
from utils.fluxo_runner import executar_fluxo_com_progresso
from utils.form_reset import render_botao_nova_consulta

st.set_page_config(page_title="Acompanhamento Obstétrico", page_icon="🤰", layout="wide")
aplicar_estilos_globais()

st.markdown("""
<div class="main-header">
    <h1>🤰 Acompanhamento Obstétrico</h1>
    <p>Risco gestacional • Exames por trimestre • Alertas de urgência • Acompanhamento</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# Seletor de paciente (sidebar)
# ========================================
contexto = render_seletor_sidebar()

prefill_nome = contexto.nome if contexto else ""
prefill_idade = contexto.idade if contexto else 28
prefill_gestacoes_ant = contexto.gestacoes if contexto else 0
prefill_partos_ant = contexto.partos_normais + contexto.partos_cesareos if contexto else 0
prefill_abortos_ant = contexto.abortos if contexto else 0
prefill_medicamentos = contexto.medicamentos_uso if contexto else ""
prefill_alergias = contexto.alergias if contexto else ""

# ========================================
# Formulário
# ========================================
with st.form("form_obstetrico"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Dados da Gestante")
        nome = st.text_input("Nome", value=prefill_nome, placeholder="Ex: Juliana Santos")
        idade = st.number_input("Idade", min_value=10, max_value=60, value=prefill_idade)
        semanas = st.number_input("Semanas de gestação", min_value=1, max_value=42, value=12)
        tipo_gestacao = st.selectbox("Tipo de gestação", ["única", "gemelar"])

    with col2:
        st.markdown("#### 📊 Histórico Obstétrico")
        gestacoes_ant = st.number_input("Gestações anteriores", min_value=0, max_value=15, value=prefill_gestacoes_ant)
        partos_ant = st.number_input("Partos anteriores", min_value=0, max_value=15, value=prefill_partos_ant)
        abortos_ant = st.number_input("Abortos anteriores", min_value=0, max_value=10, value=prefill_abortos_ant)
        grupo_sang = st.selectbox("Grupo sanguíneo", ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"])

    with col3:
        st.markdown("#### 🏥 Dados Clínicos")
        peso = st.number_input("Peso atual (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.5)
        altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.2, value=1.60, step=0.01)

        st.markdown("**Pressão arterial (mmHg)**")
        col_pas, col_pad = st.columns(2)
        with col_pas:
            pa_sistolica = st.number_input(
                "Sistólica", min_value=70, max_value=250, value=120,
                step=1, key="pa_sistolica", help="Pressão arterial sistólica em mmHg",
            )
        with col_pad:
            pa_diastolica = st.number_input(
                "Diastólica", min_value=40, max_value=160, value=80,
                step=1, key="pa_diastolica", help="Pressão arterial diastólica em mmHg",
            )
        glicemia = st.number_input("Glicemia de jejum (mg/dL)", min_value=0.0, max_value=500.0, value=85.0, step=1.0)

    # Validação inline de pressão arterial (gestantes)
    avisos_pa = []
    if pa_sistolica >= 140 or pa_diastolica >= 90:
        avisos_pa.append("⚠️ **Hipertensão na gestação** — PA ≥ 140/90 mmHg. Investigar pré-eclâmpsia.")
    elif pa_sistolica >= 130 or pa_diastolica >= 85:
        avisos_pa.append("🟡 PA limítrofe — recomenda-se aferição seriada.")
    if pa_sistolica < 90 or pa_diastolica < 60:
        avisos_pa.append("🟡 Hipotensão — avaliar sintomas (tontura, lipotimia).")
    if pa_sistolica <= pa_diastolica:
        avisos_pa.append("❌ Sistólica deve ser maior que diastólica.")
    for aviso in avisos_pa:
        st.warning(aviso)

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("#### 💊 Comorbidades e Medicações")
        comorbidades = st.multiselect("Comorbidades", [
            "Diabetes", "Hipertensão", "Hipotireoidismo", "Hipertireoidismo",
            "Epilepsia", "Cardiopatia", "Doença renal", "HIV",
            "Asma", "Depressão/Ansiedade", "Pré-eclâmpsia prévia", "Outro",
        ])
        medicamentos = st.text_area(
            "Medicamentos em uso",
            value=prefill_medicamentos,
            placeholder="Ex: Levotiroxina 50mcg, Ácido fólico",
            height=60,
        )
        if prefill_alergias and prefill_alergias != "Nenhuma":
            st.caption(f"⚠️ Alergias registradas: {prefill_alergias}")

    with col5:
        st.markdown("#### 🔍 Queixas Atuais")
        queixas = st.multiselect("Queixas da gestante", [
            "Sangramento vaginal", "Perda de líquido", "Contrações regulares",
            "Ausência de movimentos fetais", "Cefaleia intensa", "Visão turva",
            "Edema súbito", "Dor abdominal intensa", "Febre",
            "Náuseas/vômitos intensos", "Edema em membros inferiores",
            "Dor lombar", "Prurido generalizado",
        ])

    submitted = st.form_submit_button("🚀 Avaliar Gestação", use_container_width=True, type="primary")

# ========================================
# Execução do fluxo
# ========================================
if submitted:
    if not nome:
        st.warning("⚠️ Preencha o nome da gestante.")
        st.stop()
    if pa_sistolica <= pa_diastolica:
        st.error("❌ Pressão arterial inválida: sistólica deve ser maior que diastólica.")
        st.stop()

    from flows.obstetrico import criar_fluxo_obstetrico

    meds = [m.strip() for m in medicamentos.split(",") if m.strip()] if medicamentos else []

    paciente_id_str = (
        f"PAC-{contexto.paciente_id:04d}" if contexto else f"PAC-{hash(nome) % 10000:04d}"
    )

    entrada = {
        "paciente_id": paciente_id_str,
        "nome_paciente": nome,
        "idade": idade,
        "semanas_gestacao": semanas,
        "tipo_gestacao": tipo_gestacao,
        "gestacoes_anteriores": gestacoes_ant,
        "partos_anteriores": partos_ant,
        "abortos_anteriores": abortos_ant,
        "comorbidades": comorbidades,
        "medicamentos_em_uso": meds,
        "pressao_arterial": f"{int(pa_sistolica)}/{int(pa_diastolica)}",
        "peso_atual": peso,
        "altura": altura,
        "glicemia_jejum": glicemia,
        "grupo_sanguineo": grupo_sang,
        "queixas_atuais": queixas,
    }

    fluxo = criar_fluxo_obstetrico()
    resultado = executar_fluxo_com_progresso(
        fluxo,
        entrada,
        titulo="⏳ Avaliando gestação...",
        titulo_final="✅ Avaliação obstétrica concluída",
        rotulos_nos={
            "coletar_dados_gestante": "Consolidando dados da gestante (IMC, trimestre)",
            "avaliar_risco_gestacional": "Avaliando risco gestacional (LLM + regras)",
            "gerar_orientacoes": "Gerando orientações por trimestre (LLM)",
            "agendar_exames": "Agendando exames do trimestre",
            "verificar_alertas": "Verificando alertas de urgência",
            "planejar_acompanhamento": "Planejando frequência de consultas",
        },
    )

    # ========================================
    # Resultados
    # ========================================
    st.divider()
    col_titulo, col_acao = st.columns([3, 1])
    with col_titulo:
        st.markdown("## 📊 Avaliação Obstétrica")
    with col_acao:
        render_botao_nova_consulta(chave_botao="btn_nova_avaliacao_obstetrica")

    nivel = resultado.get("nivel_risco", "habitual")

    # Trimestre e risco
    tri_label = (
        "1º Trimestre" if semanas <= 13
        else "2º Trimestre" if semanas <= 27
        else "3º Trimestre"
    )

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("📅 Idade Gestacional", f"{semanas} semanas")
    with col_info2:
        st.metric("🗓️ Trimestre", tri_label)
    with col_info3:
        risco_emoji = {"alto": "🔴", "moderado": "🟡", "habitual": "🟢"}
        st.metric("⚠️ Risco Gestacional", f"{risco_emoji.get(nivel, '⚪')} {nivel.upper()}")

    # Alertas de urgência
    alertas = resultado.get("alertas_urgencia", [])
    if alertas:
        st.markdown("### 🚨 Alertas de Urgência")
        for a in alertas:
            if "EMERGÊNCIA" in a:
                st.markdown(f"""
                <div class="alerta-critico">
                    <strong>{a}</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(a)

    # Exames agendados
    st.markdown("### 🧪 Exames Agendados")
    exames = resultado.get("exames_agendados", [])
    cols_ex = st.columns(min(len(exames), 3)) if exames else []
    for i, e in enumerate(exames):
        with cols_ex[i % len(cols_ex)]:
            badge = "badge-urgente" if e.get("prioridade") == "urgente" else "badge-rotina"
            st.markdown(f"""
            <div class="info-card">
                <strong>{e['nome']}</strong>
                <span class="exame-badge {badge}">{e.get('prioridade', 'rotina').upper()}</span><br>
                <small>📅 Prazo: {e.get('prazo', 'N/A')}</small>
            </div>
            """, unsafe_allow_html=True)

    # Acompanhamento
    st.markdown("### 📋 Plano de Acompanhamento")
    acomp = resultado.get("plano_acompanhamento", {})
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        st.metric("Próxima Consulta", acomp.get("proxima_consulta", "—"))
    with col_a2:
        st.metric("Frequência", acomp.get("frequencia_consultas", "—"))
    with col_a3:
        st.metric("Encaminhamento Alto Risco", "Sim" if acomp.get("encaminhamento_alto_risco") else "Não")

    # Orientações da IA
    orient = resultado.get("orientacoes", "")
    if orient and "[LLM indisponível" not in orient:
        st.markdown("### 🤖 Orientações da IA")
        st.markdown(f'<div class="llm-card">{orient}</div>', unsafe_allow_html=True)

    exibir_disclaimer()
