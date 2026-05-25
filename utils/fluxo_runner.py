"""
Executor de fluxos LangGraph com feedback de progresso por nó.

Em vez de um spinner genérico, mostra cada nó conforme é concluído usando
`st.status` + `graph.stream()`.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import streamlit as st


def _rotulo_amigavel(no: str) -> str:
    """Converte 'identificar_sinais_alerta' → 'Identificar sinais alerta'."""
    return no.replace("_", " ").capitalize()


def executar_fluxo_com_progresso(
    grafo: Any,
    entrada: dict,
    *,
    titulo: str = "Executando fluxo clínico...",
    titulo_final: str = "Fluxo concluído",
    rotulos_nos: Optional[dict[str, str]] = None,
) -> dict:
    """
    Roda um grafo LangGraph compilado e mostra progresso por nó via st.status.

    Args:
        grafo: Grafo compilado (de `criar_fluxo_*().`)
        entrada: Estado inicial enviado ao grafo.
        titulo: Título mostrado enquanto roda.
        titulo_final: Título mostrado ao concluir.
        rotulos_nos: Mapa opcional `nome_no -> rótulo legível`.

    Returns:
        Estado final acumulado pelo grafo (mesmo formato de `graph.invoke`).
    """
    rotulos_nos = rotulos_nos or {}
    estado_final: dict = {}

    with st.status(titulo, expanded=True) as status:
        inicio = time.perf_counter()
        try:
            for evento in grafo.stream(entrada):
                # evento = {"nome_do_no": {atualizacoes}}
                for nome_no, atualizacao in evento.items():
                    rotulo = rotulos_nos.get(nome_no) or _rotulo_amigavel(nome_no)
                    elapsed = time.perf_counter() - inicio
                    st.write(f"✓ **{rotulo}** _(t={elapsed:.1f}s)_")
                    if isinstance(atualizacao, dict):
                        estado_final.update(atualizacao)
            status.update(
                label=f"{titulo_final} em {time.perf_counter() - inicio:.1f}s",
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            status.update(label=f"Falha: {exc}", state="error", expanded=True)
            raise

    # Como `stream()` devolve apenas as updates de cada nó, garantimos o estado
    # completo combinando a entrada com tudo o que foi acumulado.
    resultado = {**entrada, **estado_final}
    return resultado
