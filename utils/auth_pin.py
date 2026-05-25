"""
Helper de autenticação por PIN — UI compartilhada entre páginas.

Centraliza o estado de identidade do profissional (`identidade_verificada` e
`profissional_id`) em `st.session_state`, evitando duplicação de UI nas páginas
do Assistente Médico e da Auditoria.

A verificação real continua sendo feita por
`medical_assistant.security_protocols.verificar_identidade_profissional`.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


PROFISSIONAL_ANONIMO = "anonimo"
PROFISSIONAL_AUTENTICADO = "profissional_autenticado"


def _init_state() -> None:
    if "identidade_verificada" not in st.session_state:
        st.session_state.identidade_verificada = False
    if "profissional_id" not in st.session_state:
        st.session_state.profissional_id = PROFISSIONAL_ANONIMO


def render_login_sidebar(
    *,
    titulo: str = "🔐 Verificação de Identidade",
    descricao: str = "Obrigatória para triagem VD e dados sensíveis.",
    registrar_acesso: bool = False,
    tipo_acesso: str = "login",
) -> bool:
    """
    Renderiza o formulário de PIN na sidebar e atualiza o estado da sessão.

    Args:
        titulo: cabeçalho exibido.
        descricao: legenda abaixo do título.
        registrar_acesso: se True, registra log de acesso sensível ao autenticar.
        tipo_acesso: tipo do dado para o log (ex.: 'auditoria').

    Returns:
        True se a sessão está autenticada; False caso contrário.
    """
    _init_state()

    try:
        from medical_assistant.security_protocols import verificar_identidade_profissional
        if registrar_acesso:
            from medical_assistant.audit import registrar_acesso_sensivel
    except ImportError as exc:
        with st.sidebar:
            st.error(f"Módulo de segurança indisponível: {exc}")
        return False

    with st.sidebar:
        st.markdown(f"### {titulo}")
        if descricao:
            st.caption(descricao)

        with st.form(f"form_pin_{tipo_acesso}", clear_on_submit=False, border=False):
            pin = st.text_input(
                "PIN do profissional",
                type="password",
                key=f"_pin_input_{tipo_acesso}",
                label_visibility="collapsed",
                placeholder="PIN do profissional",
            )
            submetido = st.form_submit_button("Validar identidade", use_container_width=True)

        if submetido:
            if verificar_identidade_profissional(pin):
                st.session_state.identidade_verificada = True
                st.session_state.profissional_id = PROFISSIONAL_AUTENTICADO
                if registrar_acesso:
                    try:
                        registrar_acesso_sensivel(
                            tipo_acesso, "login",
                            profissional_id=st.session_state.profissional_id,
                        )
                    except Exception:
                        pass
                st.success("Sessão autenticada.")
            else:
                st.session_state.identidade_verificada = False
                st.session_state.profissional_id = PROFISSIONAL_ANONIMO
                st.error("PIN inválido.")

        if st.session_state.identidade_verificada:
            st.success("✅ Autenticado")
            if st.button("🚪 Encerrar sessão", use_container_width=True, key=f"_logout_{tipo_acesso}"):
                st.session_state.identidade_verificada = False
                st.session_state.profissional_id = PROFISSIONAL_ANONIMO
                st.rerun()
        else:
            st.warning("Áreas sensíveis bloqueadas")

    return st.session_state.identidade_verificada


def esta_autenticado() -> bool:
    _init_state()
    return st.session_state.identidade_verificada


def profissional_atual() -> str:
    _init_state()
    return st.session_state.profissional_id
