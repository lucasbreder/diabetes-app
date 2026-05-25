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

PROTOCOLOS DE SEGURANÇA (complementares):
• Áreas sensíveis (violência doméstica) exigem verificação de identidade do profissional.
• Dados de violência doméstica são armazenados com criptografia e acesso auditado.
• Situações de risco crítico ou emergência disparam alertas à equipe de segurança.
• Toda resposta passa por validação automática antes de ser exibida ao profissional.
===
"""

AVISO_POS_RESPOSTA = (
    "\n\n---\n"
    "⚠️ *Suporte à decisão clínica. Prescrições e diagnósticos definitivos exigem "
    "validação do profissional de saúde responsável.*"
)

_PADROES_PRESCRICAO = re.compile(
    r"(prescrev|prescriç|receita de|tome \d|tomar \d|use \d|usar \d|"
    r"administrar \d|inicie tratamento|iniciar tratamento|dosagem de|"
    r"\d+\s*mg\b|\d+\s*comprimid|\d+\s*c[áa]psul|\d+\s*gota|\d+\s*ml\b|"
    r"posologia)",
    re.IGNORECASE,
)
_PADROES_DIAGNOSTICO_DEFINITIVO = re.compile(
    r"\b(você tem|você possui|você está com|diagnóstico (é|confirmado)|"
    r"confirmad[oa] que|com certeza tem|trata-se de|é caso de|"
    r"definitivamente (tem|apresenta))\b",
    re.IGNORECASE,
)
_PADROES_VALIDACAO = re.compile(
    r"\b(validação|validar|especialista|médico responsável|"
    r"profissional de saúde|hipótese|suspeita|quadro compatível)\b",
    re.IGNORECASE,
)
_PADROES_VIOLENCIA_ENTRADA = re.compile(
    r"\b(viol[êe]ncia|agress[ãa]o|abuso|maus-tratos|"
    r"viol[êe]ncia dom[ée]stica|amea[çc]a (do parceiro|f[íi]sica))\b",
    re.IGNORECASE,
)
_PADROES_ENCAMINHAMENTO_VIOLENCIA = re.compile(
    r"\b(encaminh|assistente social|psicolog|acolhimento|"
    r"rede de prote[çc][ãa]o|180|cvl|creas|delegacia da mulher|sinan)\b",
    re.IGNORECASE,
)
_PADROES_SINTOMAS_ALARME = re.compile(
    r"\b(sangramento (intenso|abundante|profuso)|"
    r"dor (s[úu]bita|intensa|aguda|tor[áa]cica)|"
    r"febre (alta|>38|acima de 38)|emerg[êe]ncia|urg[êe]ncia imediata|"
    r"perda de consci[êe]ncia|convuls[ãa]o|dispneia)\b",
    re.IGNORECASE,
)
_PADROES_CONSULTA_PRESENCIAL = re.compile(
    r"\b(consulta presencial|atendimento presencial|"
    r"procurar (urg[êe]ncia|emerg[êe]ncia|pronto-socorro|hospital)|"
    r"samu|192|encaminhamento imediato|avalia[çc][ãa]o presencial)\b",
    re.IGNORECASE,
)
# Item 5: confidencialidade — detecta PII em respostas de contexto VD
_PADROES_PII_SENSIVEL = re.compile(
    r"(\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"          # CPF
    r"|\b\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b"           # telefone
    r"|\brua\s+[a-zà-ú]+|\bavenida\s+[a-zà-ú]+"    # endereço
    r"|\bn[úu]mero\s+\d+)",
    re.IGNORECASE,
)


def aplicar_guardrails_resposta(
    texto: str,
    mensagem_usuario: str = "",
    nivel_risco_vd: str | None = None,
    contexto_vd: bool = False,
) -> str:
    """
    Pós-valida a resposta do modelo e acrescenta avisos quando detecta
    possível violação dos 5 limites específicos de atuação.

    Args:
        texto: resposta gerada pelo LLM.
        mensagem_usuario: pergunta/contexto do profissional (entrada).
        nivel_risco_vd: risco apurado pelo WAST (baixo|moderado|alto|critico) — força
            a checagem de encaminhamento (item 3) mesmo quando a entrada não cita violência.
        contexto_vd: True quando a interação é da triagem VD ou prontuário VD — ativa
            checagem de PII na saída (item 5).
    """
    if not texto or not texto.strip():
        return texto

    avisos: list[str] = []

    if _PADROES_PRESCRICAO.search(texto) and not _PADROES_VALIDACAO.search(texto):
        avisos.append(
            "[Limite 1] Resposta pode conter indicação medicamentosa — "
            "requer validação do especialista."
        )

    if _PADROES_DIAGNOSTICO_DEFINITIVO.search(texto):
        avisos.append(
            "[Limite 2] Evite diagnóstico definitivo; reformule como hipótese "
            "clínica para o profissional avaliar."
        )

    entrada_menciona_vd = bool(
        mensagem_usuario and _PADROES_VIOLENCIA_ENTRADA.search(mensagem_usuario)
    )
    risco_vd_relevante = nivel_risco_vd in ("moderado", "alto", "critico")
    if (
        (entrada_menciona_vd or risco_vd_relevante or contexto_vd)
        and not _PADROES_ENCAMINHAMENTO_VIOLENCIA.search(texto)
    ):
        avisos.append(
            "[Limite 3] Casos suspeitos de violência devem ser encaminhados a profissionais "
            "qualificados (assistente social, psicologia, CREAS, Delegacia da Mulher, CVL 180)."
        )

    if (
        mensagem_usuario
        and _PADROES_SINTOMAS_ALARME.search(mensagem_usuario)
        and not _PADROES_CONSULTA_PRESENCIAL.search(texto)
    ):
        avisos.append(
            "[Limite 4] Sintomas alarmantes exigem recomendação explícita de "
            "consulta presencial ou urgência."
        )

    if (contexto_vd or risco_vd_relevante) and _PADROES_PII_SENSIVEL.search(texto):
        avisos.append(
            "[Limite 5] Possível dado identificável (CPF/telefone/endereço) em contexto VD — "
            "remover antes de divulgar fora do prontuário autorizado."
        )

    resultado = texto
    if avisos:
        bloco = "\n".join(f"• {a}" for a in avisos)
        resultado += f"\n\n---\n**⚠️ Validação de segurança:**\n{bloco}"

    if _PADROES_PRESCRICAO.search(texto) or _PADROES_DIAGNOSTICO_DEFINITIVO.search(texto):
        if AVISO_POS_RESPOSTA.strip() not in resultado:
            resultado += AVISO_POS_RESPOSTA

    return resultado


def stream_com_guardrails(
    chunks,
    mensagem_usuario: str = "",
    nivel_risco_vd: str | None = None,
    contexto_vd: bool = False,
):
    """Envolve um gerador de streaming e aplica guardrails ao final."""
    texto = ""
    for chunk in chunks:
        texto += chunk
        yield chunk
    final = aplicar_guardrails_resposta(
        texto,
        mensagem_usuario,
        nivel_risco_vd=nivel_risco_vd,
        contexto_vd=contexto_vd,
    )
    if len(final) > len(texto):
        yield final[len(texto):]
