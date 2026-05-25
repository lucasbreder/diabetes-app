"""
Regras de segurança e validação especializadas do Assistente Médico.
Aplicadas em todos os prompts e, quando necessário, na pós-validação das respostas.
"""

from __future__ import annotations

import re

# Bloco injetado em todos os prompts do assistente
REGRAS_SEGURANCA_PROMPT = """
=== SEGURANÇA E VALIDAÇÃO ESPECIALIZADAS (OBRIGATÓRIO) ===
LIMITES ESPECÍFICOS DE ATUAÇÃO:
• NUNCA prescrever medicações sem validação de especialista; cite apenas opções para discussão clínica com o médico responsável.
• NUNCA diagnosticar definitivamente condições sensíveis; use suspeita clínica, hipótese ou quadro compatível.
• SEMPRE encaminhar casos suspeitos de violência para profissionais qualificados (assistente social, psicologia, equipe de acolhimento, rede de proteção).
• SEMPRE sugerir consulta presencial imediata ou urgência para sintomas alarmantes.
• MANTER confidencialidade absoluta em casos de violência doméstica; não divulgar informações fora do contexto clínico autorizado.
Este sistema é suporte à decisão clínica — NÃO substitui o julgamento do profissional de saúde.
===
"""

AVISO_POS_RESPOSTA = (
    "\n\n---\n"
    "⚠️ *Suporte à decisão clínica. Prescrições e diagnósticos definitivos exigem "
    "validação do profissional de saúde responsável.*"
)

_PADROES_PRESCRICAO = re.compile(
    r"\b(prescrev|prescriç|receita de|tome \d|dosagem de|\d+\s*mg\b)\b",
    re.IGNORECASE,
)
_PADROES_DIAGNOSTICO_DEFINITIVO = re.compile(
    r"\b(você tem|você possui|diagnóstico (é|confirmado)|confirmad[oa] que|com certeza tem)\b",
    re.IGNORECASE,
)
_PADROES_VALIDACAO = re.compile(
    r"\b(validação|validar|especialista|médico responsável|profissional de saúde|hipótese|suspeita)\b",
    re.IGNORECASE,
)
_PADROES_VIOLENCIA_ENTRADA = re.compile(
    r"\b(violência|violencia|agressão|agressao|abuso|maus-tratos|violência doméstica)\b",
    re.IGNORECASE,
)
_PADROES_ENCAMINHAMENTO_VIOLENCIA = re.compile(
    r"\b(encaminh|assistente social|psicolog|acolhimento|rede de proteção|180|cvl)\b",
    re.IGNORECASE,
)
_PADROES_SINTOMAS_ALARME = re.compile(
    r"\b(sangramento (intenso|abundante)|dor (súbita|intensa|aguda)|febre (alta|>38)|emergência|urgência imediata)\b",
    re.IGNORECASE,
)
_PADROES_CONSULTA_PRESENCIAL = re.compile(
    r"\b(consulta presencial|atendimento presencial|procurar (urgência|emergência|pronto-socorro)|samu)\b",
    re.IGNORECASE,
)


def aplicar_guardrails_resposta(
    texto: str,
    mensagem_usuario: str = "",
) -> str:
    """
    Pós-valida a resposta do modelo e acrescenta avisos quando detecta
    possível violação das regras de segurança.
    """
    if not texto or not texto.strip():
        return texto

    avisos: list[str] = []

    if _PADROES_PRESCRICAO.search(texto) and not _PADROES_VALIDACAO.search(texto):
        avisos.append(
            "A resposta pode conter indicação medicamentosa — requer validação do especialista."
        )

    if _PADROES_DIAGNOSTICO_DEFINITIVO.search(texto):
        avisos.append(
            "Evite diagnóstico definitivo; reformule como hipótese clínica para o profissional avaliar."
        )

    if (
        mensagem_usuario
        and _PADROES_VIOLENCIA_ENTRADA.search(mensagem_usuario)
        and not _PADROES_ENCAMINHAMENTO_VIOLENCIA.search(texto)
    ):
        avisos.append(
            "Casos suspeitos de violência devem ser encaminhados a profissionais qualificados "
            "(assistente social, psicologia, rede de proteção — CVL 180)."
        )

    if (
        mensagem_usuario
        and _PADROES_SINTOMAS_ALARME.search(mensagem_usuario)
        and not _PADROES_CONSULTA_PRESENCIAL.search(texto)
    ):
        avisos.append(
            "Sintomas alarmantes exigem recomendação explícita de consulta presencial ou urgência."
        )

    resultado = texto
    if avisos:
        bloco = "\n".join(f"• {a}" for a in avisos)
        resultado += f"\n\n---\n**⚠️ Validação de segurança:**\n{bloco}"

    if _PADROES_PRESCRICAO.search(texto) or _PADROES_DIAGNOSTICO_DEFINITIVO.search(texto):
        if AVISO_POS_RESPOSTA.strip() not in resultado:
            resultado += AVISO_POS_RESPOSTA

    return resultado


def stream_com_guardrails(chunks, mensagem_usuario: str = ""):
    """Envolve um gerador de streaming e aplica guardrails ao final."""
    texto = ""
    for chunk in chunks:
        texto += chunk
        yield chunk
    final = aplicar_guardrails_resposta(texto, mensagem_usuario)
    if len(final) > len(texto):
        yield final[len(texto):]
