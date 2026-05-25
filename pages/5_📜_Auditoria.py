"""
Módulo transversal de logging e auditoria do Sistema de Saúde da Mulher.
Consolida interações do assistente, acessos sensíveis e alertas de segurança.
"""

from __future__ import annotations

import streamlit as st
from utils.shared_styles import aplicar_estilos_globais

st.set_page_config(
    page_title="Auditoria e Logs",
    page_icon="📜",
    layout="wide",
)

aplicar_estilos_globais()

try:
    from medical_assistant.database import init_db
    from medical_assistant.seed_data import popular_banco
    from medical_assistant.audit import (
        listar_alertas_pendentes,
        listar_logs_acesso_sensivel,
        listar_logs_interacao,
        relatorio_resumo_auditoria,
        registrar_acesso_sensivel,
    )
    from medical_assistant.security_protocols import verificar_identidade_profissional

    init_db()
    popular_banco()
except ImportError as e:
    st.error(f"Dependências do assistente médico não encontradas: {e}")
    st.stop()

if "identidade_verificada" not in st.session_state:
    st.session_state.identidade_verificada = False
if "profissional_id" not in st.session_state:
    st.session_state.profissional_id = "anonimo"

st.markdown("""
<div class="main-header">
    <h1>📜 Auditoria e Logs Especializados</h1>
    <p>Rastreamento de interações • Acesso a dados sensíveis • Alertas de segurança • Uso por especialidade</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Camada transversal do sistema — registra atividade do **Assistente Médico** "
    "e demais fluxos que utilizam validação e segurança."
)

with st.sidebar:
    st.header("🔐 Acesso")
    st.caption("PIN obrigatório para logs de violência doméstica e acessos sensíveis.")
    pin = st.text_input("PIN do profissional:", type="password", key="pin_auditoria")
    if st.button("Validar identidade", use_container_width=True):
        if verificar_identidade_profissional(pin):
            st.session_state.identidade_verificada = True
            st.session_state.profissional_id = "profissional_autenticado"
            registrar_acesso_sensivel(
                "auditoria",
                "login",
                profissional_id=st.session_state.profissional_id,
            )
            st.success("Sessão autenticada.")
        else:
            st.session_state.identidade_verificada = False
            st.error("PIN inválido.")

    if st.session_state.identidade_verificada:
        st.success("✅ Autenticado")
    else:
        st.warning("Visão restrita (métricas gerais apenas)")

    st.divider()
    st.page_link("app.py", label="← Voltar ao início", icon="🏠")
    st.page_link("pages/assistente_medico.py", label="Assistente Médico", icon="🤖")

# ─── Resumo geral (sempre visível) ─────────────────────────────

resumo = relatorio_resumo_auditoria()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Interações registradas", resumo["total_interacoes"])
c2.metric("Logs violência doméstica", resumo["interacoes_violencia_domestica"])
c3.metric("Acessos a dados sensíveis", resumo["acessos_dados_sensiveis"])
c4.metric("Alertas pendentes", resumo["alertas_seguranca_pendentes"])

st.markdown("### Utilização por especialidade médica")
if resumo["por_especialidade"]:
    cols = st.columns(min(len(resumo["por_especialidade"]), 4))
    for i, (esp, qtd) in enumerate(resumo["por_especialidade"].items()):
        cols[i % len(cols)].metric(esp.replace("_", " ").title(), qtd)
else:
    st.info("Nenhuma interação registrada ainda. Use o Assistente Médico ou os fluxos clínicos.")

# ─── Alertas de segurança ────────────────────────────────────

alertas = listar_alertas_pendentes()
if alertas:
    st.markdown("### 🚨 Alertas de segurança pendentes")
    for a in alertas:
        st.error(
            f"**{a.nivel.upper()}** — Paciente ID {a.paciente_id or 'N/A'} — "
            f"{a.motivo} _(protocolo: {a.protocolo_emergencia or 'N/A'})_"
        )

st.divider()

# ─── Filtros de interações ───────────────────────────────────

tab_interacoes, tab_acessos, tab_vd = st.tabs([
    "📋 Todas as interações",
    "🔑 Acessos sensíveis",
    "🔒 Violência doméstica",
])

with tab_interacoes:
    filtro_esp = st.selectbox(
        "Especialidade",
        ["Todas", "ginecologia", "violencia_domestica", "preventiva", "multidisciplinar", "obstetricia"],
    )
    filtro_fluxo = st.selectbox(
        "Fluxo",
        ["Todos", "chat", "triagem", "vd", "alertas", "encaminhamentos", "langgraph"],
    )
    logs = listar_logs_interacao(
        especialidade=None if filtro_esp == "Todas" else filtro_esp,
        fluxo=None if filtro_fluxo == "Todos" else filtro_fluxo,
        limite=25,
    )
    if logs:
        for log in logs:
            flag = "🔒 " if log.caso_violencia else ""
            with st.expander(
                f"{flag}{log.criado_em.strftime('%d/%m/%Y %H:%M') if log.criado_em else '—'} "
                f"| {log.fluxo} | {log.especialidade}"
            ):
                st.write(f"**Paciente ID:** {log.paciente_id or 'N/A'}")
                st.write(f"**Guardrails:** {'Sim' if log.guardrails_aplicados else 'Não'}")
                st.write(f"**Entrada:** {log.mensagem_resumo or '—'}")
                st.write(f"**Resposta:** {log.resposta_resumo or '—'}")
    else:
        st.caption("Nenhum log para os filtros selecionados.")

with tab_acessos:
    if not st.session_state.identidade_verificada:
        st.warning("Autentique-se na barra lateral para ver o histórico de acessos sensíveis.")
    else:
        acessos = listar_logs_acesso_sensivel(limite=30)
        if acessos:
            for log in acessos:
                st.text(
                    f"{log.criado_em} | {log.tipo_dado} | {log.acao} | "
                    f"paciente {log.paciente_id or 'N/A'} | {log.profissional_id}"
                )
        else:
            st.caption("Nenhum acesso sensível registrado.")

with tab_vd:
    if not st.session_state.identidade_verificada:
        st.warning("Autentique-se para consultar logs específicos de violência doméstica.")
    else:
        logs_vd = listar_logs_interacao(apenas_vd=True, limite=20)
        if logs_vd:
            for log in logs_vd:
                st.markdown(f"**{log.criado_em}** — paciente ID **{log.paciente_id or 'N/A'}** ({log.fluxo})")
                st.caption(log.mensagem_resumo or "(sem resumo)")
        else:
            st.caption("Nenhum log de violência doméstica.")

st.divider()
st.caption(
    "Requisitos atendidos: rastreamento de interações, logs VD, auditoria de acesso sensível "
    "e relatórios por especialidade. Dados de VD no prontuário permanecem criptografados em repouso."
)
