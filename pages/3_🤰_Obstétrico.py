import streamlit as st
from utils.shared_styles import aplicar_estilos_globais, card_urgencia, exibir_disclaimer

st.set_page_config(page_title="Acompanhamento Obstétrico", page_icon="🤰", layout="wide")
aplicar_estilos_globais()

st.markdown("""
<div class="main-header">
    <h1>🤰 Acompanhamento Obstétrico</h1>
    <p>Risco gestacional • Exames por trimestre • Alertas de urgência • Acompanhamento</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# Formulário
# ========================================
with st.form("form_obstetrico"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Dados da Gestante")
        nome = st.text_input("Nome", placeholder="Ex: Juliana Santos")
        idade = st.number_input("Idade", min_value=10, max_value=60, value=28)
        semanas = st.number_input("Semanas de gestação", min_value=1, max_value=42, value=12)
        tipo_gestacao = st.selectbox("Tipo de gestação", ["única", "gemelar"])

    with col2:
        st.markdown("#### 📊 Histórico Obstétrico")
        gestacoes_ant = st.number_input("Gestações anteriores", min_value=0, max_value=15, value=0)
        partos_ant = st.number_input("Partos anteriores", min_value=0, max_value=15, value=0)
        abortos_ant = st.number_input("Abortos anteriores", min_value=0, max_value=10, value=0)
        grupo_sang = st.selectbox("Grupo sanguíneo", ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"])

    with col3:
        st.markdown("#### 🏥 Dados Clínicos")
        peso = st.number_input("Peso atual (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.5)
        altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.2, value=1.60, step=0.01)
        pa = st.text_input("Pressão arterial", placeholder="Ex: 120/80")
        glicemia = st.number_input("Glicemia de jejum (mg/dL)", min_value=0.0, max_value=500.0, value=85.0, step=1.0)

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("#### 💊 Comorbidades e Medicações")
        comorbidades = st.multiselect("Comorbidades", [
            "Diabetes", "Hipertensão", "Hipotireoidismo", "Hipertireoidismo",
            "Epilepsia", "Cardiopatia", "Doença renal", "HIV",
            "Asma", "Depressão/Ansiedade", "Pré-eclâmpsia prévia", "Outro",
        ])
        medicamentos = st.text_area("Medicamentos em uso", placeholder="Ex: Levotiroxina 50mcg, Ácido fólico", height=60)

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

    from flows.obstetrico import criar_fluxo_obstetrico

    meds = [m.strip() for m in medicamentos.split(",") if m.strip()] if medicamentos else []

    entrada = {
        "paciente_id": f"PAC-{hash(nome) % 10000:04d}",
        "nome_paciente": nome,
        "idade": idade,
        "semanas_gestacao": semanas,
        "tipo_gestacao": tipo_gestacao,
        "gestacoes_anteriores": gestacoes_ant,
        "partos_anteriores": partos_ant,
        "abortos_anteriores": abortos_ant,
        "comorbidades": comorbidades,
        "medicamentos_em_uso": meds,
        "pressao_arterial": pa or "N/A",
        "peso_atual": peso,
        "altura": altura,
        "glicemia_jejum": glicemia,
        "grupo_sanguineo": grupo_sang,
        "queixas_atuais": queixas,
    }

    with st.spinner("⏳ Avaliando gestação..."):
        fluxo = criar_fluxo_obstetrico()
        resultado = fluxo.invoke(entrada)

    # ========================================
    # Resultados
    # ========================================
    st.divider()
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
