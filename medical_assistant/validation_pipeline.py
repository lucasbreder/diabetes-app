"""
Orquestra validação, auditoria, alertas e explicabilidade (item 4 – seções 2–4).
"""

from __future__ import annotations

from typing import Optional

from .audit import (
    registrar_acesso_sensivel,
    registrar_alerta_seguranca,
    registrar_interacao,
    registrar_log_violencia,
)
from .explainability import MetadadosExplicacao, construir_explicacao, formatar_bloco_explicabilidade
from .security_protocols import (
    avaliar_situacao_emergencia,
    validar_estabilidade_resposta,
)

_CATEGORIA_POR_FLUXO = {
    "vd": "violencia_domestica",
    "triagem_vd": "violencia_domestica",
    "obstetrico": "obstetricia",
    "obstetricia": "obstetricia",
    "prevencao": "rastreamento_cancer",
    "alertas": "rastreamento_cancer",
    "triagem": None,
    "encaminhamentos": None,
    "chat": None,
    "langgraph": None,
}


def _autobuscar_protocolos(fluxo: str, mensagem_usuario: str) -> str:
    """Recupera protocolos indexados da base quando o caller não os passou.
    Resultado formatado como '[Título – Fonte]\\nResumo...' para o extrator de fontes."""
    try:
        from .database import buscar_protocolos
    except Exception:
        return ""

    categoria = _CATEGORIA_POR_FLUXO.get(fluxo)
    resultados = []
    if categoria:
        resultados = buscar_protocolos(categoria=categoria)
    if not resultados and mensagem_usuario:
        resultados = buscar_protocolos(termo=mensagem_usuario)

    if not resultados:
        return ""
    partes = []
    for p in resultados[:2]:
        resumo = (p.conteudo or "")[:400]
        if len(p.conteudo or "") > 400:
            resumo += "..."
        partes.append(f"[{p.titulo} – {p.fonte}]\n{resumo}")
    return "\n\n".join(partes)


def processar_resposta_final(
    resposta_bruta: str,
    *,
    mensagem_usuario: str = "",
    paciente_id: Optional[int] = None,
    fluxo: str = "chat",
    especialidade: str = "ginecologia",
    protocolos_contexto: str = "",
    contexto_paciente: str = "",
    nivel_risco_vd: Optional[str] = None,
    incluir_explicabilidade: bool = True,
    profissional_id: str = "sessao_streamlit",
) -> tuple[str, Optional[MetadadosExplicacao]]:
    """
    Valida resposta do LLM, dispara alertas, registra auditoria e monta explicabilidade.
    """
    emergencia = avaliar_situacao_emergencia(
        mensagem_usuario,
        nivel_risco_vd=nivel_risco_vd,
    )
    if emergencia.acionar_alerta:
        registrar_alerta_seguranca(
            paciente_id=paciente_id,
            nivel=emergencia.nivel,
            motivo=emergencia.mensagem_equipe,
            protocolo_emergencia=emergencia.protocolo,
        )

    caso_vd = fluxo in ("vd", "triagem_vd") or bool(
        nivel_risco_vd and nivel_risco_vd != "baixo"
    )
    resposta_validada, ajustes = validar_estabilidade_resposta(
        resposta_bruta,
        mensagem_usuario,
        nivel_risco_vd=nivel_risco_vd,
        contexto_vd=caso_vd,
    )

    meta = None
    if incluir_explicabilidade:
        protocolos_efetivos = protocolos_contexto
        if not protocolos_efetivos:
            protocolos_efetivos = _autobuscar_protocolos(fluxo, mensagem_usuario)
        meta = construir_explicacao(
            resposta_validada,
            protocolos_contexto=protocolos_efetivos,
            contexto_paciente=contexto_paciente,
            mensagem_usuario=mensagem_usuario,
            ajustes_seguranca=ajustes,
            fluxo=fluxo,
        )

    registrar_interacao(
        fluxo=fluxo,
        especialidade=especialidade,
        mensagem=mensagem_usuario,
        resposta=resposta_validada,
        paciente_id=paciente_id,
        guardrails_aplicados=bool(ajustes),
        caso_violencia=caso_vd,
        metadados={
            "confianca": meta.confianca_pct if meta else None,
            "emergencia": emergencia.nivel,
            "ajustes": ajustes,
        },
    )

    if caso_vd and paciente_id:
        registrar_log_violencia(
            paciente_id=paciente_id,
            acao="interacao_assistente",
            profissional_id=profissional_id,
            detalhes={"fluxo": fluxo, "nivel_risco": nivel_risco_vd},
        )

    texto_final = resposta_validada
    if meta and incluir_explicabilidade:
        texto_final += formatar_bloco_explicabilidade(meta)

    return texto_final, meta


def stream_com_processamento(chunks, **kwargs):
    """Envolve streaming do LLM e aplica validação/auditoria/explicabilidade ao final."""
    texto = ""
    for chunk in chunks:
        texto += chunk
        yield chunk
    final, _ = processar_resposta_final(texto, **kwargs)
    if len(final) > len(texto):
        yield final[len(texto):]
