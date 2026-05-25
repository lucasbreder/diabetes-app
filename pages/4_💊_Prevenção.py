import streamlit as st
from utils.shared_styles import aplicar_estilos_globais, exibir_disclaimer

st.set_page_config(page_title="Prevenção", page_icon="💊", layout="wide")
aplicar_estilos_globais()

st.markdown("""
<div class="main-header">
    <h1>💊 Prevenção e Rastreamento</h1>
    <p>Exames devidos • Orientações preventivas • Agendamento automático • Lembretes</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# Formulário
# ========================================
with st.form("form_prevencao"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 Dados da Paciente")
        nome = st.text_input("Nome", placeholder="Ex: Fernanda Costa")
        idade = st.number_input("Idade", min_value=10, max_value=100, value=45)
        historico_pessoal = st.multiselect("Histórico pessoal", [
            "Hipertensão", "Diabetes tipo 2", "Obesidade", "Tabagismo",
            "Depressão", "Osteoporose", "Câncer prévio", "Doença cardiovascular",
        ])
        comorbidades = st.multiselect("Comorbidades ativas", [
            "Hipertensão", "Diabetes", "Dislipidemia", "Hipotireoidismo",
            "Obesidade", "Depressão/Ansiedade",
        ])

    with col2:
        st.markdown("#### 👨‍👩‍👧 Histórico Familiar")
        hist_familiar = st.multiselect("Doenças na família (1º grau)", [
            "Câncer de mama", "Câncer de colo uterino", "Câncer de ovário",
            "Câncer colorretal", "Diabetes tipo 2", "Hipertensão",
            "Doença cardiovascular", "Osteoporose", "Alzheimer",
        ])

    st.markdown("#### 📅 Últimos Exames Realizados")
    col3, col4, col5 = st.columns(3)
    with col3:
        ultimo_papa = st.text_input("Último Papanicolau", placeholder="Ex: 2023-03-10 ou 'nunca'")
        ultima_mamo = st.text_input("Última Mamografia", placeholder="Ex: 2022-08-20 ou 'nunca'")
    with col4:
        ultima_densito = st.text_input("Última Densitometria", placeholder="Ex: 2024-01-15 ou 'nunca'")
        ultimo_checkup = st.text_input("Último Check-up geral", placeholder="Ex: 2024-06-01")
    with col5:
        vacinas = st.multiselect("Vacinas em dia", [
            "Gripe 2024", "Gripe 2025", "COVID-19", "HPV",
            "Hepatite B", "dTpa", "Febre amarela",
        ])

    st.markdown("#### 🏃‍♀️ Hábitos de Vida")
    col6, col7, col8, col9 = st.columns(4)
    with col6:
        tabagismo = st.selectbox("Tabagismo", [
            "Nunca fumou", "Ex-fumante", "Fumante ativa",
        ])
    with col7:
        etilismo = st.selectbox("Consumo de álcool", [
            "Não consome", "Social/ocasional", "Regular", "Excessivo",
        ])
    with col8:
        atividade = st.selectbox("Atividade física", [
            "Sedentária", "Leve (1-2x/sem)", "Moderada (3-4x/sem)", "Intensa (5+x/sem)",
        ])
    with col9:
        alimentacao = st.selectbox("Alimentação", [
            "Balanceada", "Rica em ultraprocessados", "Vegetariana/vegana", "Restritiva",
        ])

    submitted = st.form_submit_button("🚀 Analisar Prevenção", use_container_width=True, type="primary")

# ========================================
# Execução do fluxo
# ========================================
if submitted:
    if not nome:
        st.warning("⚠️ Preencha o nome da paciente.")
        st.stop()

    from flows.prevencao import criar_fluxo_prevencao

    entrada = {
        "paciente_id": f"PAC-{hash(nome) % 10000:04d}",
        "nome_paciente": nome,
        "idade": idade,
        "sexo": "F",
        "historico_pessoal": historico_pessoal,
        "historico_familiar": hist_familiar,
        "ultimo_papanicolau": ultimo_papa or "nunca",
        "ultima_mamografia": ultima_mamo or "nunca",
        "ultima_densitometria": ultima_densito or "nunca",
        "ultimo_check_up": ultimo_checkup or "nunca",
        "vacinas_em_dia": vacinas,
        "habitos": {
            "tabagismo": tabagismo,
            "etilismo": etilismo,
            "atividade_fisica": atividade,
            "alimentacao": alimentacao,
        },
        "comorbidades": comorbidades,
    }

    with st.spinner("⏳ Analisando perfil preventivo..."):
        fluxo = criar_fluxo_prevencao()
        resultado = fluxo.invoke(entrada)

    # ========================================
    # Resultados
    # ========================================
    st.divider()

    exames = resultado.get("exames_devidos", [])
    agendamentos = resultado.get("agendamentos", [])
    lembretes = resultado.get("lembretes", [])

    # Métricas
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("🧪 Exames Devidos", len(exames))
    with col_m2:
        st.metric("📅 Agendamentos Criados", len(agendamentos))
    with col_m3:
        st.metric("🔔 Lembretes Programados", len(lembretes))

    # Exames devidos
    if exames:
        st.markdown("### 🧪 Exames em Atraso ou Nunca Realizados")
        for e in exames:
            status_emoji = "🔴" if e["status"] == "NUNCA REALIZADO" else "🟡"
            badge_class = "badge-urgente" if e["status"] == "NUNCA REALIZADO" else "badge-rotina"
            st.markdown(f"""
            <div class="info-card">
                <strong>{status_emoji} {e['exame']}</strong>
                <span class="exame-badge {badge_class}">{e['status']}</span><br>
                <small>📊 Protocolo: {e['intervalo']} | 📝 {e.get('observacao', '')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Todos os exames de rastreamento estão em dia!")

    # Agendamentos
    if agendamentos:
        st.markdown("### 📅 Agendamentos Automáticos")
        cols = st.columns(min(len(agendamentos), 3))
        for i, a in enumerate(agendamentos):
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div class="info-card" style="text-align:center;">
                    <strong>{a['exame']}</strong><br>
                    <span style="font-size:1.3rem;font-weight:700;color:#1565c0;">{a['data_sugerida']}</span><br>
                    <small>Prioridade: {a['prioridade']}</small><br>
                    <small>📋 {a.get('preparo', '')}</small>
                </div>
                """, unsafe_allow_html=True)

    # Lembretes
    if lembretes:
        st.markdown("### 🔔 Lembretes Programados")
        lem_agendamento = [l for l in lembretes if l["tipo"] == "agendamento"]
        lem_preventivo = [l for l in lembretes if l["tipo"] == "preventivo"]

        if lem_agendamento:
            st.markdown("**📅 Lembretes de agendamento:**")
            for l in lem_agendamento:
                st.info(f"{l['mensagem']} — Envio: {l['data_envio']} ({l['canal']})")

        if lem_preventivo:
            st.markdown("**🔮 Lembretes futuros (próximos rastreamentos):**")
            for l in lem_preventivo:
                st.success(f"{l['mensagem']} — Envio: {l['data_envio']} ({l['canal']})")

    # Orientações da IA
    orient = resultado.get("orientacoes_preventivas", "")
    if orient and "[LLM indisponível" not in orient:
        st.markdown("### 🤖 Orientações Preventivas da IA")
        st.markdown(f'<div class="llm-card">{orient}</div>', unsafe_allow_html=True)

    exibir_disclaimer()
