from datetime import date

import streamlit as st
from utils.shared_styles import aplicar_estilos_globais, exibir_disclaimer
from utils.paciente_selector import render_seletor_sidebar
from utils.fluxo_runner import executar_fluxo_com_progresso

st.set_page_config(page_title="Prevenção", page_icon="💊", layout="wide")
aplicar_estilos_globais()

st.markdown("""
<div class="main-header">
    <h1>💊 Prevenção e Rastreamento</h1>
    <p>Exames devidos • Orientações preventivas • Agendamento automático • Lembretes</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# Seletor de paciente (sidebar)
# ========================================
contexto = render_seletor_sidebar()

prefill_nome = contexto.nome if contexto else ""
prefill_idade = contexto.idade if contexto else 45


def _campo_data_exame(label: str, *, data_existente: date | None, key: str) -> str:
    """Renderiza um par (date_input + checkbox 'Nunca realizado').
    Retorna ISO ('YYYY-MM-DD') ou 'nunca'."""
    col_check, col_data = st.columns([1, 2])
    with col_check:
        nunca = st.checkbox(
            "Nunca realizado",
            value=(data_existente is None),
            key=f"{key}_nunca",
        )
    with col_data:
        data_default = data_existente or date.today()
        if nunca:
            st.date_input(label, value=data_default, disabled=True, key=f"{key}_data")
            return "nunca"
        valor = st.date_input(
            label,
            value=data_default,
            max_value=date.today(),
            format="DD/MM/YYYY",
            key=f"{key}_data",
        )
        return valor.isoformat()


# ========================================
# Formulário
# ========================================
with st.form("form_prevencao"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 Dados da Paciente")
        nome = st.text_input("Nome", value=prefill_nome, placeholder="Ex: Fernanda Costa")
        idade = st.number_input("Idade", min_value=10, max_value=100, value=prefill_idade)
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
    st.caption("Marque _Nunca realizado_ se a paciente nunca fez o exame; caso contrário, informe a data.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Papanicolau**")
        ultimo_papa = _campo_data_exame(
            "Data",
            data_existente=contexto.exames_por_tipo.get("papanicolau") if contexto else None,
            key="papa",
        )
        st.markdown("**Mamografia**")
        ultima_mamo = _campo_data_exame(
            "Data",
            data_existente=contexto.exames_por_tipo.get("mamografia") if contexto else None,
            key="mamo",
        )
    with col4:
        st.markdown("**Densitometria óssea**")
        ultima_densito = _campo_data_exame(
            "Data",
            data_existente=contexto.exames_por_tipo.get("densitometria") if contexto else None,
            key="densito",
        )
        st.markdown("**Check-up geral**")
        ultimo_checkup = _campo_data_exame(
            "Data",
            data_existente=None,
            key="checkup",
        )

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

    paciente_id_str = (
        f"PAC-{contexto.paciente_id:04d}" if contexto else f"PAC-{hash(nome) % 10000:04d}"
    )

    entrada = {
        "paciente_id": paciente_id_str,
        "nome_paciente": nome,
        "idade": idade,
        "sexo": "F",
        "historico_pessoal": historico_pessoal,
        "historico_familiar": hist_familiar,
        "ultimo_papanicolau": ultimo_papa,
        "ultima_mamografia": ultima_mamo,
        "ultima_densitometria": ultima_densito,
        "ultimo_check_up": ultimo_checkup,
        "vacinas_em_dia": vacinas,
        "habitos": {
            "tabagismo": tabagismo,
            "etilismo": etilismo,
            "atividade_fisica": atividade,
            "alimentacao": alimentacao,
        },
        "comorbidades": comorbidades,
    }

    fluxo = criar_fluxo_prevencao()
    resultado = executar_fluxo_com_progresso(
        fluxo,
        entrada,
        titulo="⏳ Analisando perfil preventivo...",
        titulo_final="✅ Análise preventiva concluída",
        rotulos_nos={
            "analisar_historico": "Analisando histórico e fatores de risco (LLM)",
            "identificar_exames_devidos": "Identificando exames de rastreamento devidos",
            "gerar_orientacoes_preventivas": "Gerando orientações personalizadas (LLM)",
            "agendar_automaticamente": "Criando agendamentos automáticos",
            "gerar_lembretes": "Programando lembretes SMS/WhatsApp",
        },
    )

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
