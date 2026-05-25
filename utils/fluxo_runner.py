"""
Executor de fluxos LangGraph com feedback de progresso por nó.

Em vez de um spinner genérico, mostra cada nó conforme é concluído usando
`st.status` + `graph.stream()`. Inclui também um helper para envolver streams
de LLM (token a token) em um `st.status` consistente.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional

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


def stream_llm_com_status(
    gerador: Iterable[str],
    *,
    titulo: str = "Consultando LLM…",
    etapa_preparacao: Optional[str] = None,
    etapa_llm: str = "🤖 Gerando resposta",
    etapa_validacao: str = "🛡️ Validando resposta (guardrails + auditoria)",
    placeholder: Optional[Any] = None,
    colapsar_ao_finalizar: bool = True,
) -> str:
    """
    Envolve um stream de chunks de LLM em um `st.status` consistente.

    O passo a passo (preparação → LLM → validação) fica DENTRO do status
    (colapsável). O texto da resposta é renderizado FORA, em um placeholder
    próprio, garantindo que a resposta permaneça visível após o status colapsar.

    Args:
        gerador: Iterável que produz pedaços (str) da resposta do LLM.
        titulo: Título inicial do status.
        etapa_preparacao: Linha exibida antes do streaming (None = omite).
        etapa_llm: Rótulo da fase de geração.
        etapa_validacao: Rótulo da fase de pós-processamento.
        placeholder: `st.empty()` opcional onde escrever o texto. Se None, cria.
        colapsar_ao_finalizar: Se True, fecha o status ao terminar com sucesso.

    Returns:
        Texto completo acumulado (já com avisos pós-validação se a função
        chamadora aplicar guardrails dentro do gerador).
    """
    texto = ""

    # Cria o status (mantém referência para .update no final)
    status = st.status(titulo, expanded=True)
    # Placeholder do texto FORA do status — permanece visível após colapsar
    if placeholder is None:
        placeholder = st.empty()

    inicio = time.perf_counter()
    try:
        with status:
            if etapa_preparacao:
                st.write(etapa_preparacao)
            st.write(etapa_llm)

        for chunk in gerador:
            texto += chunk
            placeholder.markdown(texto + "▌")
        placeholder.markdown(texto)

        with status:
            st.write(etapa_validacao)

        status.update(
            label=f"✅ Concluído em {time.perf_counter() - inicio:.1f}s — clique para ver etapas",
            state="complete",
            expanded=not colapsar_ao_finalizar,
        )
    except Exception as exc:
        status.update(label=f"❌ Falha: {exc}", state="error", expanded=True)
        raise

    return texto
