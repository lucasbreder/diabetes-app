"""
Explicabilidade contextualizada (item 4 – seção 4).
Fontes, raciocínio clínico, confiança e lacunas de informação.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetadadosExplicacao:
    fontes: list[str] = field(default_factory=list)
    raciocinio: list[str] = field(default_factory=list)
    confianca: str = "moderada"  # baixa | moderada | alta
    confianca_pct: int = 65
    informacao_adicional: list[str] = field(default_factory=list)
    ajustes_seguranca: list[str] = field(default_factory=list)


_PADROES_LACUNA = re.compile(
    r"\b(não informado|nao informado|desconhecido|a definir|sem dados|"
    r"faltam dados|informação insuficiente|necessário mais)\b",
    re.IGNORECASE,
)
_PADROES_HIPOTESE = re.compile(
    r"\b(hipótese|hipotese|suspeita|compatível|pode indicar|sugere que)\b",
    re.IGNORECASE,
)
_PADROES_URGENCIA = re.compile(
    r"\b(emergência|urgência|urgente|imediato|samu|pronto-socorro)\b",
    re.IGNORECASE,
)

_FONTE_FALLBACK = "Conhecimento geral — FEBRASGO / Ministério da Saúde (sem protocolo indexado)"
_FONTE_BASE_INTERNA = "Base interna de protocolos hospitalares"


def _inferir_confianca(
    resposta: str,
    fontes: list[str],
    guardrails: bool,
    contexto_paciente: str = "",
) -> tuple[str, int]:
    """Score 15-95; só conta como protocolo real fontes específicas (não-fallback)."""
    score = 50

    fontes_reais = [
        f for f in fontes
        if f not in (_FONTE_FALLBACK, _FONTE_BASE_INTERNA)
        and not f.startswith("Conhecimento geral")
    ]
    if fontes_reais:
        score += min(30, 20 + 5 * (len(fontes_reais) - 1))
    if _PADROES_HIPOTESE.search(resposta):
        score -= 10
    if _PADROES_URGENCIA.search(resposta):
        score += 5
    if guardrails:
        score -= 15
    if _PADROES_LACUNA.search(resposta):
        score -= 20
    if not contexto_paciente or "Nenhuma paciente" in contexto_paciente:
        score -= 10

    score = max(15, min(95, score))
    if score >= 75:
        return "alta", score
    if score >= 50:
        return "moderada", score
    return "baixa", score


def extrair_fontes_protocolos(protocolos_texto: str) -> list[str]:
    """Extrai cabeçalhos de protocolo no formato '[Título – Fonte]' (com travessão U+2013).
    Blocos sem o separador são ignorados para não capturar listas internas como '[ASC-US]'."""
    fontes = []
    for bloco in re.findall(r"\[([^\]]+)\]", protocolos_texto or ""):
        if " – " not in bloco:
            continue
        titulo, fonte = bloco.split(" – ", 1)
        fontes.append(f"{titulo.strip()} ({fonte.strip()})")
    if not fontes and protocolos_texto and "Nenhum protocolo" not in protocolos_texto:
        fontes.append(_FONTE_BASE_INTERNA)
    if not fontes:
        fontes.append(_FONTE_FALLBACK)
    return fontes[:5]


def construir_explicacao(
    resposta: str,
    *,
    protocolos_contexto: str = "",
    contexto_paciente: str = "",
    mensagem_usuario: str = "",
    ajustes_seguranca: Optional[list[str]] = None,
    fluxo: str = "chat",
) -> MetadadosExplicacao:
    fontes = extrair_fontes_protocolos(protocolos_contexto)
    if fluxo == "vd":
        fontes.insert(0, "Protocolo institucional — Violência contra a mulher (Lei 11.340/2006, SINAN)")
    elif fluxo == "triagem":
        fontes.insert(0, "Protocolo de triagem ginecológica/obstétrica")

    raciocinio = []
    if _PADROES_HIPOTESE.search(resposta):
        raciocinio.append("A orientação usa linguagem de hipótese clínica, não diagnóstico definitivo.")
    if _PADROES_URGENCIA.search(resposta):
        raciocinio.append("Foi identificada menção a urgência ou emergência na conduta sugerida.")
    if contexto_paciente and "EXAMES ATRASADOS" in contexto_paciente:
        raciocinio.append("Exames preventivos em atraso foram considerados no contexto da paciente.")
    if "violência" in (mensagem_usuario or "").lower() or fluxo == "vd":
        raciocinio.append(
            "Contexto sensível (violência doméstica): encaminhamento a equipe qualificada é prioritário."
        )
    if not raciocinio:
        raciocinio.append(
            "Raciocínio baseado em sintomas/queixa informados e protocolos recuperados da base."
        )

    info_extra = []
    if not contexto_paciente or "Nenhuma paciente" in contexto_paciente:
        info_extra.append("Selecione ou informe dados da paciente para maior precisão.")
    if _PADROES_LACUNA.search(resposta) or _PADROES_LACUNA.search(contexto_paciente or ""):
        info_extra.append("Há lacunas nos dados clínicos — complementar anamnese e exames.")
    if fluxo == "triagem" and not mensagem_usuario:
        info_extra.append("Detalhar duração e intensidade dos sintomas melhora a classificação.")

    nivel, pct = _inferir_confianca(
        resposta,
        fontes,
        bool(ajustes_seguranca),
        contexto_paciente=contexto_paciente,
    )

    return MetadadosExplicacao(
        fontes=fontes,
        raciocinio=raciocinio,
        confianca=nivel,
        confianca_pct=pct,
        informacao_adicional=info_extra,
        ajustes_seguranca=ajustes_seguranca or [],
    )


def formatar_bloco_explicabilidade(meta: MetadadosExplicacao) -> str:
    """Markdown para exibição na interface."""
    linhas = ["\n\n---\n### 📊 Explicabilidade contextualizada\n"]

    linhas.append("**Fontes consultadas:**")
    for f in meta.fontes:
        linhas.append(f"- {f}")

    linhas.append("\n**Raciocínio clínico (resumo):**")
    for r in meta.raciocinio:
        linhas.append(f"- {r}")

    emoji_conf = {"alta": "🟢", "moderada": "🟡", "baixa": "🟠"}.get(meta.confianca, "⚪")
    linhas.append(
        f"\n**Nível de confiança nas sugestões:** {emoji_conf} "
        f"{meta.confianca.capitalize()} ({meta.confianca_pct}%)"
    )
    linhas.append(
        "_Confiança reflete completude dos dados e aderência a protocolos indexados — "
        "não substitui avaliação do profissional._"
    )

    if meta.informacao_adicional:
        linhas.append("\n**⚠️ Informação adicional necessária:**")
        for i in meta.informacao_adicional:
            linhas.append(f"- {i}")

    if meta.ajustes_seguranca:
        linhas.append("\n**Validação de segurança aplicada:**")
        for a in meta.ajustes_seguranca:
            linhas.append(f"- {a}")

    return "\n".join(linhas)
