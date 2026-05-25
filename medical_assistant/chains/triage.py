"""
Chain de triagem automática de sintomas ginecológicos/obstétricos.
Classifica urgência e sugere conduta inicial baseada nos sintomas relatados.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from ..safety import REGRAS_SEGURANCA_PROMPT, aplicar_guardrails_resposta, stream_com_guardrails

PROMPT_TRIAGEM = PromptTemplate.from_template(
    """Você é uma enfermeira obstetra especializada em triagem clínica de saúde da mulher.
Sua função é classificar a urgência clínica com base nos sintomas relatados e orientar o profissional de saúde sobre a conduta inicial.

{regras_seguranca}

CLASSIFICAÇÕES DE URGÊNCIA:
- EMERGENCIA: Risco imediato de vida -> encaminhar imediatamente para servico de emergência.
- URGENCIA: Avaliação médica nas próximas 2-4 horas.
- PRIORITARIO: Consulta necessária em 24-48 horas.
- ELETIVO: Consulta de rotina.

Responda sempre em português brasileiro. Seja direta, técnica e empática.

Dados da paciente para triagem:
Sintomas relatados: {sintomas}
Duração dos sintomas: {duracao}
Intensidade (0-10): {intensidade}
Histórico relevante: {historico}
Dados adicionais: {contexto_paciente}

Elabore a triagem clínica com as seções: Classificação de Urgência, Sintomas em Análise, Sinais de Alarme, Conduta Sugerida, Encaminhamentos Indicados.

Triagem:"""
)


def executar_triagem_sintomas(
    sintomas: list[str],
    duracao: str = "Não informado",
    intensidade: int | None = None,
    historico: str = "Não informado",
    contexto_paciente: str = "",
    modelo: str = "llama3:latest",
) -> str:
    """
    Executa a chain de triagem de sintomas e retorna a classificação clínica.

    Args:
        sintomas: Lista de sintomas relatados.
        duracao: Duração dos sintomas.
        intensidade: Intensidade (0-10).
        historico: Histórico clínico relevante.
        contexto_paciente: Dados formatados da paciente (prontuário, etc.).
        modelo: Modelo Ollama a usar.

    Returns:
        Texto com a triagem clínica estruturada.
    """
    llm = OllamaLLM(model=modelo, temperature=0.2)
    chain = PROMPT_TRIAGEM | llm | StrOutputParser()

    entrada = {
        "sintomas": "; ".join(sintomas) if sintomas else "Não informados",
        "duracao": duracao,
        "intensidade": f"{intensidade}/10" if intensidade is not None else "Não informada",
        "historico": historico,
        "contexto_paciente": contexto_paciente or "Sem dados adicionais",
        "regras_seguranca": REGRAS_SEGURANCA_PROMPT,
    }
    texto_sintomas = entrada["sintomas"]
    resposta = chain.invoke(entrada)
    return aplicar_guardrails_resposta(resposta, texto_sintomas)


def stream_triagem_sintomas(
    sintomas: list[str],
    duracao: str = "Não informado",
    intensidade: int | None = None,
    historico: str = "Não informado",
    contexto_paciente: str = "",
    modelo: str = "llama3:latest",
):
    """Versão streaming da triagem de sintomas para uso com Streamlit."""
    llm = OllamaLLM(model=modelo, temperature=0.2)
    chain = PROMPT_TRIAGEM | llm | StrOutputParser()

    texto_sintomas = "; ".join(sintomas) if sintomas else "Não informados"
    stream = chain.stream({
        "sintomas": texto_sintomas,
        "duracao": duracao,
        "intensidade": f"{intensidade}/10" if intensidade is not None else "Não informada",
        "historico": historico,
        "contexto_paciente": contexto_paciente or "Sem dados adicionais",
        "regras_seguranca": REGRAS_SEGURANCA_PROMPT,
    })
    return stream_com_guardrails(stream, texto_sintomas)
