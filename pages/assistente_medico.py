"""
Página Streamlit: Assistente Médica Especializada em Saúde da Mulher
Interface completa com chat contextualizado, triagem, alertas e encaminhamentos.
"""

from __future__ import annotations

import json
from datetime import date

import streamlit as st

# ─────────────────────────────────────────────
# Configuração da Página
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Assistente Médica – Saúde da Mulher",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.header-assistente {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
}
.header-assistente h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.header-assistente p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

.patient-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.patient-card h4 { margin: 0 0 0.5rem 0; color: #1e293b; font-size: 1rem; }
.patient-card p { margin: 0.2rem 0; color: #475569; font-size: 0.88rem; }

.alert-card {
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.alert-atrasado {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #ef4444;
}
.alert-alterado {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
}
.risk-critico { background: #7f1d1d; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; }
.risk-alto { background: #dc2626; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; }
.risk-moderado { background: #d97706; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; }
.risk-baixo { background: #059669; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; }

.confidential-banner {
    background: #1e1b4b;
    color: #c7d2fe;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    font-size: 0.82rem;
    margin-bottom: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Imports com tratamento de erro
# ─────────────────────────────────────────────

@st.cache_resource
def _carregar_dependencias():
    try:
        from medical_assistant.database import init_db, listar_pacientes
        from medical_assistant.seed_data import popular_banco
        init_db()
        popular_banco()
        return True, None
    except ImportError as e:
        return False, str(e)


deps_ok, deps_erro = _carregar_dependencias()

if not deps_ok:
    st.error(f"""
    **Dependências não encontradas.** Execute:
    ```
    pip install langchain langchain-ollama langchain-community sqlalchemy
    ```
    Erro: `{deps_erro}`
    """)
    st.stop()

from medical_assistant.database import (
    buscar_exames,
    buscar_exames_atrasados,
    buscar_exames_alterados,
    buscar_ciclos_menstruais,
    buscar_paciente,
    buscar_prontuarios,
    buscar_triagens_vd,
    listar_pacientes,
    registrar_triagem_vd,
)
from medical_assistant.chains.triage import stream_triagem_sintomas
from medical_assistant.chains.alerts import stream_alertas_exames, gerar_alertas_exames
from medical_assistant.chains.dv_screening import (
    PERGUNTAS_WAST,
    stream_triagem_violencia,
    calcular_risco_wast,
)
from medical_assistant.chains.referrals import stream_encaminhamentos
from medical_assistant.pipeline import AssistenteMedico
from medical_assistant.audit import listar_alertas_pendentes, registrar_acesso_sensivel
from medical_assistant.security_protocols import verificar_identidade_profissional


# ─────────────────────────────────────────────
# Estado da Sessão
# ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "paciente_id": None,
        "assistente": None,
        "mensagens_chat": [],
        "modelo": "llama3:latest",
        "identidade_verificada": False,
        "profissional_id": "anonimo",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("""
<div class="header-assistente">
    <h1>🏥 Assistente Médica Especializada</h1>
    <p>Saúde da Mulher · Ginecologia · Obstetrícia · Saúde Reprodutiva · Powered by LangChain + Ollama</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar – Seleção de Paciente
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("👩‍⚕️ Paciente em Atendimento")

    pacientes = listar_pacientes()

    opcoes = {f"{p.nome} (ID {p.id})": p.id for p in pacientes}
    opcoes_lista = ["— Selecionar paciente —"] + list(opcoes.keys())

    selecao = st.selectbox("Selecione a paciente:", opcoes_lista)

    if selecao != "— Selecionar paciente —":
        novo_id = opcoes[selecao]
        if novo_id != st.session_state.paciente_id:
            st.session_state.paciente_id = novo_id
            st.session_state.mensagens_chat = []
            if st.session_state.assistente:
                st.session_state.assistente.definir_paciente(novo_id)

    st.divider()

    st.subheader("⚙️ Configurações")
    modelo_selecionado = st.selectbox(
        "Modelo Ollama:",
        ["llama3:latest", "llama3.1:8b", "mistral:7b", "gemma2:2b"],
        index=0,
    )
    if modelo_selecionado != st.session_state.modelo:
        st.session_state.modelo = modelo_selecionado
        st.session_state.assistente = None  # força recriação

    st.divider()
    st.subheader("🔐 Verificação de identidade")
    st.caption("Obrigatória para triagem VD e dados sensíveis.")
    pin = st.text_input("PIN do profissional:", type="password", key="pin_profissional")
    if st.button("Validar acesso", use_container_width=True):
        if verificar_identidade_profissional(pin):
            st.session_state.identidade_verificada = True
            st.session_state.profissional_id = "profissional_autenticado"
            st.success("Identidade verificada.")
        else:
            st.session_state.identidade_verificada = False
            st.error("PIN inválido.")
    if st.session_state.identidade_verificada:
        st.success("✅ Sessão autenticada")
    else:
        st.warning("Áreas sensíveis bloqueadas")

    alertas_seg = listar_alertas_pendentes(limite=5)
    if alertas_seg:
        st.divider()
        st.subheader("🚨 Alertas de segurança")
        for a in alertas_seg:
            st.error(f"**{a.nivel.upper()}** — {a.motivo[:80]}…")

    st.page_link("pages/5_📜_Auditoria.py", label="📜 Auditoria completa", icon="📊")

    st.divider()
    st.caption("⚠️ Este sistema auxilia profissionais de saúde e **não substitui** o julgamento clínico.")
    st.caption("🔒 Dados de VD criptografados em repouso (Fernet).")

    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.mensagens_chat = []
        if st.session_state.assistente:
            st.session_state.assistente.limpar_historico()
        st.rerun()


# ─────────────────────────────────────────────
# Dados da Paciente Selecionada
# ─────────────────────────────────────────────

pac_id = st.session_state.paciente_id

if pac_id:
    paciente = buscar_paciente(pac_id)
    if paciente:
        hoje = date.today()
        idade = hoje.year - paciente.data_nascimento.year - (
            (hoje.month, hoje.day) < (paciente.data_nascimento.month, paciente.data_nascimento.day)
        )
        ultimo_pront = buscar_prontuarios(pac_id)
        up = ultimo_pront[0] if ultimo_pront else None

        col_info, col_alertas = st.columns([2, 1])

        with col_info:
            st.markdown(f"""
            <div class="patient-card">
                <h4>👤 {paciente.nome}</h4>
                <p>📅 {paciente.data_nascimento.strftime('%d/%m/%Y')} &nbsp;|&nbsp; {idade} anos</p>
                {"<p>🤰 G" + str(up.gestacoes) + "P" + str(up.partos_normais + up.partos_cesareos) + "A" + str(up.abortos) + "</p>" if up else ""}
                {"<p>💊 " + (up.metodo_contraceptivo or "Sem método informado") + "</p>" if up else ""}
                {"<p>⚠️ Alergias: " + up.alergias + "</p>" if up and up.alergias and up.alergias != 'Nenhuma' else ""}
            </div>
            """, unsafe_allow_html=True)

        with col_alertas:
            atrasados = buscar_exames_atrasados(pac_id)
            alterados = buscar_exames_alterados(pac_id)
            if atrasados:
                st.warning(f"⚠️ {len(atrasados)} exame(s) atrasado(s)")
            if alterados:
                st.warning(f"🔬 {len(alterados)} resultado(s) alterado(s)")
            if not atrasados and not alterados:
                st.success("✅ Exames em dia")

            vd_hist = buscar_triagens_vd(pac_id)
            if vd_hist and vd_hist[0].nivel_risco in ("alto", "critico"):
                st.error(f"🔴 Risco VD: {vd_hist[0].nivel_risco.upper()}")

    st.divider()
else:
    st.info("👈 Selecione uma paciente na barra lateral para começar o atendimento.")


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab_chat, tab_triagem, tab_alertas, tab_vd, tab_encaminhamentos = st.tabs([
    "💬 Chat",
    "🩺 Triagem de Sintomas",
    "📋 Alertas de Exames",
    "🔒 Triagem VD",
    "🔀 Encaminhamentos",
])


# ─── TAB 1: CHAT ───────────────────────────────

with tab_chat:
    if not pac_id:
        st.info("Selecione uma paciente ou faça perguntas gerais sobre saúde da mulher.")

    # Inicializar assistente
    if st.session_state.assistente is None:
        with st.spinner("Iniciando assistente médica..."):
            try:
                st.session_state.assistente = AssistenteMedico(
                    paciente_id=pac_id,
                    modelo=st.session_state.modelo,
                )
            except Exception as e:
                st.error(f"Erro ao iniciar assistente: {e}")
                st.stop()

    # Exibir histórico
    for msg in st.session_state.mensagens_chat:
        with st.chat_message(msg["role"], avatar="🏥" if msg["role"] == "assistant" else "👩‍⚕️"):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Pergunte sobre a paciente, sintomas, protocolos..."):
        st.session_state.mensagens_chat.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👩‍⚕️"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🏥"):
            with st.spinner("Consultando base de dados clínica..."):
                try:
                    resposta_completa = ""
                    placeholder = st.empty()

                    for chunk in st.session_state.assistente.stream_chat(prompt):
                        resposta_completa += chunk
                        placeholder.markdown(resposta_completa + "▌")

                    placeholder.markdown(resposta_completa)
                    st.session_state.mensagens_chat.append({
                        "role": "assistant",
                        "content": resposta_completa,
                    })
                except Exception as e:
                    erro = f"⚠️ Erro ao processar: {e}\n\nVerifique se o Ollama está rodando: `ollama serve`"
                    st.error(erro)
                    st.session_state.mensagens_chat.append({
                        "role": "assistant",
                        "content": erro,
                    })


# ─── TAB 2: TRIAGEM DE SINTOMAS ────────────────

with tab_triagem:
    st.subheader("🩺 Triagem Automática de Sintomas")

    col1, col2 = st.columns([1, 1])

    with col1:
        sintomas_texto = st.text_area(
            "Sintomas relatados (um por linha):",
            placeholder="Dor pélvica\nSangramento irregular\nNáusea\nFebbre",
            height=150,
        )
        duracao = st.text_input("Duração dos sintomas:", placeholder="Ex: 3 dias, 2 semanas")
        intensidade = st.slider("Intensidade da dor (0-10):", 0, 10, 0)
        historico = st.text_area(
            "Histórico clínico relevante:",
            placeholder="SOP diagnosticada, cirurgia prévia, etc.",
            height=100,
        )

    with col2:
        if pac_id:
            pront = buscar_prontuarios(pac_id)
            if pront:
                up = pront[0]
                st.info(
                    f"**Contexto automático da paciente:**\n"
                    f"G{up.gestacoes}P{up.partos_normais+up.partos_cesareos}A{up.abortos} | "
                    f"{up.metodo_contraceptivo or 'Sem contraceptivo'}\n"
                    f"Alergias: {up.alergias or 'Nenhuma'}"
                )

        st.markdown("**Guia rápido de sintomas de alarme:**")
        st.markdown("""
        - 🔴 Sangramento vaginal abundante (encharca absorvente em <1h)
        - 🔴 Dor pélvica súbita e intensa
        - 🔴 Febre >38°C com dor pélvica
        - 🔴 Ausência de movimento fetal (gestantes)
        - 🟠 Corrimento com odor intenso + febre
        - 🟠 Massa pélvica palpável
        """)

    if st.button("🔍 Executar Triagem", type="primary", use_container_width=True):
        sintomas_lista = [s.strip() for s in sintomas_texto.splitlines() if s.strip()]
        if not sintomas_lista:
            st.warning("Informe pelo menos um sintoma.")
        else:
            contexto = ""
            if pac_id:
                pront = buscar_prontuarios(pac_id)
                if pront:
                    up = pront[0]
                    contexto = (
                        f"G{up.gestacoes}P{up.partos_normais+up.partos_cesareos}A{up.abortos}, "
                        f"{up.metodo_contraceptivo or 'sem contraceptivo'}, "
                        f"alergias: {up.alergias or 'nenhuma'}"
                    )

            st.divider()
            with st.spinner("Analisando sintomas..."):
                resultado_placeholder = st.empty()
                texto_acumulado = ""
                try:
                    for chunk in stream_triagem_sintomas(
                        sintomas=sintomas_lista,
                        duracao=duracao or "Não informado",
                        intensidade=intensidade if intensidade > 0 else None,
                        historico=historico or "Não informado",
                        contexto_paciente=contexto,
                        modelo=st.session_state.modelo,
                    ):
                        texto_acumulado += chunk
                        resultado_placeholder.markdown(texto_acumulado + "▌")
                    resultado_placeholder.markdown(texto_acumulado)
                except Exception as e:
                    st.error(f"Erro: {e}. Verifique se o Ollama está rodando.")


# ─── TAB 3: ALERTAS DE EXAMES ──────────────────

with tab_alertas:
    st.subheader("📋 Alertas de Exames Preventivos")

    if not pac_id:
        st.info("Selecione uma paciente para verificar alertas de exames.")
    else:
        col_resumo, col_historico = st.columns([1, 1])

        with col_resumo:
            atrasados = buscar_exames_atrasados(pac_id)
            alterados = buscar_exames_alterados(pac_id)

            if atrasados:
                st.markdown("### 🔴 Exames Atrasados")
                for e in atrasados:
                    dias = (date.today() - e.proximo_previsto).days
                    st.markdown(
                        f'<div class="alert-card alert-atrasado">'
                        f'<strong>{e.tipo_exame.upper()}</strong><br>'
                        f'Previsto em: {e.proximo_previsto.strftime("%d/%m/%Y")} '
                        f'(<span style="color:#dc2626">⚠️ {dias} dias em atraso</span>)'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ Nenhum exame com prazo vencido")

            if alterados:
                st.markdown("### 🟡 Resultados Alterados")
                for e in alterados:
                    st.markdown(
                        f'<div class="alert-card alert-alterado">'
                        f'<strong>{e.tipo_exame.upper()}</strong> ({e.data_realizacao.strftime("%d/%m/%Y") if e.data_realizacao else "N/A"})<br>'
                        f'{(e.resultado or "")[:200]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with col_historico:
            st.markdown("### 📅 Histórico Completo")
            todos_exames = buscar_exames(pac_id)
            if todos_exames:
                for e in todos_exames:
                    icone = "✅" if not e.resultado_alterado else "⚠️"
                    with st.expander(f"{icone} {e.tipo_exame.upper()} – {e.data_realizacao or 'Não realizado'}"):
                        st.write(f"**Resultado:** {e.resultado or 'Não informado'}")
                        st.write(f"**Próximo previsto:** {e.proximo_previsto or 'Não agendado'}")
                        st.write(f"**Laboratório:** {e.laboratorio or 'Não informado'}")
                        st.write(f"**Solicitante:** {e.medico_solicitante or 'Não informado'}")

        if atrasados or alterados:
            st.divider()
            if st.button("🤖 Gerar Plano de Ação com IA", type="primary"):
                with st.spinner("Gerando plano de ação..."):
                    texto_placeholder = st.empty()
                    texto_acum = ""
                    try:
                        for chunk in stream_alertas_exames(pac_id, modelo=st.session_state.modelo):
                            texto_acum += chunk
                            texto_placeholder.markdown(texto_acum + "▌")
                        texto_placeholder.markdown(texto_acum)
                    except Exception as e:
                        st.error(f"Erro: {e}")


# ─── TAB 4: TRIAGEM DE VIOLÊNCIA DOMÉSTICA ─────

with tab_vd:
    st.markdown('<div class="confidential-banner">🔒 ÁREA CONFIDENCIAL – Acesso restrito ao profissional de saúde responsável</div>', unsafe_allow_html=True)

    acesso_vd = st.session_state.identidade_verificada

    if not acesso_vd:
        st.warning(
            "Verifique sua identidade na barra lateral (PIN do profissional) "
            "para acessar a triagem de violência doméstica."
        )
    else:
        if pac_id:
            registrar_acesso_sensivel(
                "violencia_domestica",
                "leitura_aba_vd",
                paciente_id=pac_id,
                profissional_id=st.session_state.profissional_id,
            )

    st.subheader("Triagem de Violência Doméstica (WAST adaptado)")

    if acesso_vd:
        st.info(
            "**Aplicação do instrumento WAST** (Woman Abuse Screening Tool) – 8 perguntas.\n\n"
            "Conduzir em ambiente privativo, **sem acompanhante presente**.\n"
            "A notificação compulsória ao SINAN é **obrigatória** quando confirmada violência (Lei 11.340/2006)."
        )

    if not acesso_vd:
        st.caption("Formulário WAST disponível após autenticação.")
    else:
        if pac_id:
            p = buscar_paciente(pac_id)
            nome_para_relatorio = p.nome if p else f"Paciente #{pac_id}"
        else:
            nome_para_relatorio = st.text_input(
                "Nome da paciente:", placeholder="Informe o nome para o relatório"
            )

        st.markdown("### Perguntas da Triagem")
        st.caption("Selecione a resposta para cada pergunta com base na entrevista com a paciente.")

        pontuacoes = []
        opcoes_resposta = {"Não": 0, "Às vezes": 1, "Sim": 2}

        for i, pergunta in enumerate(PERGUNTAS_WAST):
            resposta = st.radio(
                f"**Q{i+1}.** {pergunta}",
                options=list(opcoes_resposta.keys()),
                horizontal=True,
                key=f"wast_q{i}",
            )
            pontuacoes.append(opcoes_resposta[resposta])

        total_wast, nivel = calcular_risco_wast(pontuacoes)

        cores_nivel = {"baixo": "🟢", "moderado": "🟡", "alto": "🟠", "critico": "🔴"}
        st.markdown(
            f"**Pontuação atual:** {total_wast}/16 &nbsp; | &nbsp; "
            f"Risco: {cores_nivel.get(nivel, '')} **{nivel.upper()}**"
        )

        sinais_fisicos_texto = st.text_area(
            "Sinais físicos observados (opcional):",
            placeholder="Hematomas, lacerações, cicatrizes suspeitas, etc.",
            height=80,
        )
        obs_clinicas = st.text_area(
            "Observações clínicas confidenciais:",
            height=80,
        )

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("📋 Gerar Relatório Confidencial", type="primary", use_container_width=True):
                if not nome_para_relatorio:
                    st.warning("Informe o nome da paciente.")
                else:
                    sinais = [s.strip() for s in sinais_fisicos_texto.splitlines() if s.strip()]
                    st.divider()

                    nivel_resultado, stream = stream_triagem_violencia(
                        nome_paciente=nome_para_relatorio,
                        pontuacoes_wast=pontuacoes,
                        sinais_fisicos=sinais if sinais else None,
                        observacoes=obs_clinicas,
                        modelo=st.session_state.modelo,
                    )

                    cores_bg = {
                        "baixo": "#f0fdf4", "moderado": "#fffbeb",
                        "alto": "#fff7ed", "critico": "#fef2f2",
                    }
                    st.markdown(
                        f'<div style="background:{cores_bg.get(nivel_resultado, "#f8fafc")};'
                        f'border-radius:10px;padding:1rem;border-left:4px solid #6366f1;">',
                        unsafe_allow_html=True,
                    )
                    placeholder = st.empty()
                    texto_acum = ""
                    try:
                        for chunk in stream:
                            texto_acum += chunk
                            placeholder.markdown(texto_acum + "▌")
                        placeholder.markdown(texto_acum)
                    except Exception as e:
                        st.error(f"Erro: {e}")
                    st.markdown("</div>", unsafe_allow_html=True)

        with col_btn2:
            if st.button("💾 Salvar Triagem no Prontuário", use_container_width=True):
                if not pac_id:
                    st.warning("Selecione uma paciente para salvar o registro.")
                else:
                    indicadores_salvos = []
                    opcoes_map = {0: "não", 1: "às vezes", 2: "sim"}
                    for i, (p, sc) in enumerate(zip(PERGUNTAS_WAST, pontuacoes)):
                        if sc > 0:
                            indicadores_salvos.append(f"Q{i+1}: {p} → {opcoes_map[sc]}")

                    try:
                        registrar_triagem_vd(
                            paciente_id=pac_id,
                            nivel_risco=nivel,
                            indicadores=json.dumps(indicadores_salvos, ensure_ascii=False),
                            protocolo_acionado=nivel in ("alto", "critico"),
                            observacoes=obs_clinicas,
                            profissional_id=st.session_state.profissional_id,
                        )
                        st.success("✅ Triagem salva (dados criptografados no banco).")
                    except Exception as e:
                        st.error(f"Erro ao salvar: {e}")

        if pac_id:
            hist_vd = buscar_triagens_vd(pac_id)
            if hist_vd:
                with st.expander("📂 Histórico de triagens anteriores"):
                    for t in hist_vd:
                        cores_nivel_html = {
                            "baixo": "#059669", "moderado": "#d97706",
                            "alto": "#dc2626", "critico": "#7f1d1d",
                        }
                        cor = cores_nivel_html.get(t.nivel_risco, "#6b7280")
                        st.markdown(
                            f"**{t.data_triagem}** – "
                            f'<span style="color:{cor};font-weight:600">{t.nivel_risco.upper()}</span>'
                            f" | Protocolo acionado: {'Sim' if t.protocolo_acionado else 'Não'}",
                            unsafe_allow_html=True,
                        )


# ─── TAB 5: ENCAMINHAMENTOS ────────────────────

with tab_encaminhamentos:
    st.subheader("🔀 Sugestão de Encaminhamentos Multidisciplinares")

    col_enc1, col_enc2 = st.columns([1, 1])

    with col_enc1:
        queixas_enc = st.text_area(
            "Queixas principais:",
            placeholder="Dor pélvica crônica, sangramento irregular...",
            height=100,
        )
        diagnostico_enc = st.text_area(
            "Diagnósticos / Suspeitas diagnósticas:",
            placeholder="Suspeita de endometriose, SOP confirmada...",
            height=100,
        )
        fatores_risco_enc = st.text_area(
            "Fatores de risco identificados:",
            placeholder="Tabagismo, obesidade, histórico familiar...",
            height=80,
        )

    with col_enc2:
        contexto_enc = ""
        if pac_id:
            p = buscar_paciente(pac_id)
            pront = buscar_prontuarios(pac_id)
            atrasados_enc = buscar_exames_atrasados(pac_id)
            alterados_enc = buscar_exames_alterados(pac_id)

            if p and pront:
                up = pront[0]
                contexto_enc = (
                    f"Paciente: {p.nome}\n"
                    f"Histórico: G{up.gestacoes}P{up.partos_normais+up.partos_cesareos}A{up.abortos}\n"
                    f"Contraceptivo: {up.metodo_contraceptivo or 'N/A'}\n"
                    f"Medicamentos: {up.medicamentos_uso or 'Nenhum'}\n"
                    f"Alergias: {up.alergias or 'Nenhuma'}"
                )

            exames_alt_texto = "\n".join(
                f"• {e.tipo_exame}: {(e.resultado or '')[:100]}" for e in alterados_enc
            ) if alterados_enc else "Nenhum"

            st.info(f"**Contexto automático carregado:**\n{contexto_enc}")
        else:
            contexto_enc = st.text_area(
                "Contexto da paciente (se não selecionada):",
                height=150,
                placeholder="Informe dados relevantes da paciente...",
            )
            exames_alt_texto = "Não informado"

    if st.button("🤖 Gerar Sugestão de Encaminhamentos", type="primary", use_container_width=True):
        if not queixas_enc.strip():
            st.warning("Informe as queixas principais.")
        else:
            st.divider()
            with st.spinner("Analisando quadro clínico..."):
                placeholder = st.empty()
                texto_acum = ""
                try:
                    for chunk in stream_encaminhamentos(
                        contexto_paciente=contexto_enc or "Não informado",
                        queixas=queixas_enc,
                        diagnosticos=diagnostico_enc or "A definir",
                        exames_alterados=exames_alt_texto,
                        fatores_risco=fatores_risco_enc or "Não identificados",
                        modelo=st.session_state.modelo,
                    ):
                        texto_acum += chunk
                        placeholder.markdown(texto_acum + "▌")
                    placeholder.markdown(texto_acum)
                except Exception as e:
                    st.error(f"Erro: {e}. Verifique se o Ollama está rodando.")

    st.divider()
    st.caption("📞 **Recursos de apoio:** CVL 180 (Central da Mulher) | SAMU 192 | Ligue 100 (violência infanto-juvenil)")
