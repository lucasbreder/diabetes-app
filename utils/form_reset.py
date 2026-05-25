"""
Helpers para reset de formulários Streamlit entre execuções.

Permite que páginas de fluxo ofereçam um botão "Nova consulta" que limpa
todos os widgets sem afetar o estado global (paciente selecionada,
autenticação, modelo, etc.).
"""

from __future__ import annotations

from typing import Iterable, Optional

import streamlit as st


# Chaves de estado que NUNCA devem ser limpas pelo "Nova consulta"
CHAVES_GLOBAIS_PADRAO = frozenset({
    # Seletor de paciente compartilhado
    "paciente_id_global",
    "_paciente_selectbox",
    # Autenticação
    "identidade_verificada",
    "profissional_id",
    # Assistente Médico
    "assistente",
    "mensagens_chat",
    "modelo",
})


def render_botao_nova_consulta(
    *,
    label: str = "🔄 Nova consulta",
    chave_botao: str = "btn_nova_consulta",
    preservar: Optional[Iterable[str]] = None,
    prefixos_preservar: Optional[Iterable[str]] = None,
    ajuda: str = "Limpa o formulário e os resultados para iniciar uma nova consulta.",
) -> bool:
    """
    Renderiza um botão "Nova consulta" e limpa o estado da página quando clicado.

    Args:
        label: rótulo do botão.
        chave_botao: key única do botão (importante quando há vários na mesma página).
        preservar: conjunto adicional de chaves a manter (somadas às globais).
        prefixos_preservar: chaves cujo nome comece com qualquer um destes prefixos
            também são mantidas (ex.: '_pin_').
        ajuda: tooltip do botão.

    Returns:
        True se o botão foi clicado (a página já terá feito rerun e este retorno
        raramente é útil).
    """
    preservar_set = set(CHAVES_GLOBAIS_PADRAO)
    if preservar:
        preservar_set.update(preservar)

    prefixos = tuple(prefixos_preservar or ()) + (
        "_paciente_",   # seletor de paciente
        "_pin_",        # PIN compartilhado
        "_logout_",     # botão de logout
        "form_pin_",    # form do PIN
        chave_botao,    # o próprio botão
    )

    clicado = st.button(label, key=chave_botao, type="secondary", help=ajuda)
    if not clicado:
        return False

    for chave in list(st.session_state.keys()):
        if chave in preservar_set:
            continue
        if chave.startswith(prefixos):
            continue
        del st.session_state[chave]

    st.rerun()
    return True  # nunca alcançado, mas mantém a assinatura
