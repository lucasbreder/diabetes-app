"""
Chain de sugestão de encaminhamentos multidisciplinares.
Baseado no contexto clínico da paciente, sugere especialidades e serviços.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from ..safety import REGRAS_SEGURANCA_PROMPT
from ..validation_pipeline import processar_resposta_final, stream_com_processamento

PROMPT_ENCAMINHAMENTOS = PromptTemplate.from_template(
    """Você é uma coordenadora de cuidados clínicos especializada na rede de saúde da mulher no Brasil.
Sugira encaminhamentos multidisciplinares adequados com base no quadro clínico da paciente.
Considere SUS e rede privada. Justifique cada encaminhamento. Priorize por urgência clínica.
Responda sempre em português brasileiro.

{regras_seguranca}

Perfil clínico da paciente:
{contexto_paciente}

Queixas Principais: {queixas}
Diagnósticos / Suspeitas: {diagnosticos}
Exames Alterados: {exames_alterados}
Fatores de Risco: {fatores_risco}

Elabore: Encaminhamentos Recomendados (especialidade, prioridade, justificativa, o que solicitar), Orientações Pós-Consulta, Sinais de Alarme, Próximos Passos.

Encaminhamentos:"""
)

PROMPT_ORIENTACOES_POS_CONSULTA = PromptTemplate.from_template(
    """Você é uma médica ginecologista gerando orientações personalizadas para a paciente ao final da consulta.
Use linguagem clara, acessível e empática. Seja específica e prática. Responda em português brasileiro.

{regras_seguranca}

Paciente: {nome_paciente}
Diagnóstico / Motivo da consulta: {diagnostico}
Medicações prescritas: {medicacoes}
Procedimentos realizados: {procedimentos}
Exames solicitados: {exames_solicitados}

Inclua: o que esperar nos próximos dias, como tomar os medicamentos (se houver), cuidados especiais e restrições, quando retornar ou buscar urgência, contatos úteis.
Máximo 15 linhas.

Orientações:"""
)


def sugerir_encaminhamentos(
    contexto_paciente: str,
    queixas: str,
    diagnosticos: str = "A definir",
    exames_alterados: str = "Nenhum",
    fatores_risco: str = "Nenhum identificado",
    modelo: str = "llama3:latest",
) -> str:
    """
    Sugere encaminhamentos multidisciplinares com base no quadro clínico.

    Args:
        contexto_paciente: Dados formatados da paciente.
        queixas: Queixas principais relatadas.
        diagnosticos: Diagnósticos ou suspeitas diagnósticas.
        exames_alterados: Exames com resultados alterados.
        fatores_risco: Fatores de risco identificados.
        modelo: Modelo Ollama a usar.

    Returns:
        Texto com encaminhamentos e orientações.
    """
    llm = OllamaLLM(model=modelo, temperature=0.3)
    chain = PROMPT_ENCAMINHAMENTOS | llm | StrOutputParser()

    resposta_bruta = chain.invoke({
        "contexto_paciente": contexto_paciente,
        "queixas": queixas,
        "diagnosticos": diagnosticos,
        "exames_alterados": exames_alterados,
        "fatores_risco": fatores_risco,
        "regras_seguranca": REGRAS_SEGURANCA_PROMPT,
    })
    resposta, _ = processar_resposta_final(
        resposta_bruta,
        mensagem_usuario=queixas,
        fluxo="encaminhamentos",
        especialidade="multidisciplinar",
        contexto_paciente=contexto_paciente,
    )
    return resposta


def gerar_orientacoes_pos_consulta(
    nome_paciente: str,
    diagnostico: str,
    medicacoes: str = "Nenhuma prescrita",
    procedimentos: str = "Nenhum",
    exames_solicitados: str = "Nenhum",
    modelo: str = "llama3:latest",
) -> str:
    """Gera orientações pós-consulta personalizadas para a paciente."""
    llm = OllamaLLM(model=modelo, temperature=0.4)
    chain = PROMPT_ORIENTACOES_POS_CONSULTA | llm | StrOutputParser()

    bruto = chain.invoke({
        "nome_paciente": nome_paciente,
        "diagnostico": diagnostico,
        "medicacoes": medicacoes,
        "procedimentos": procedimentos,
        "exames_solicitados": exames_solicitados,
        "regras_seguranca": REGRAS_SEGURANCA_PROMPT,
    })
    texto, _ = processar_resposta_final(
        bruto,
        mensagem_usuario=diagnostico,
        fluxo="orientacoes_pos_consulta",
        especialidade="ginecologia",
    )
    return texto


def stream_encaminhamentos(
    contexto_paciente: str,
    queixas: str,
    diagnosticos: str = "A definir",
    exames_alterados: str = "Nenhum",
    fatores_risco: str = "Nenhum identificado",
    modelo: str = "llama3:latest",
):
    """Versão streaming para uso com Streamlit."""
    llm = OllamaLLM(model=modelo, temperature=0.3)
    chain = PROMPT_ENCAMINHAMENTOS | llm | StrOutputParser()

    stream = chain.stream({
        "contexto_paciente": contexto_paciente,
        "queixas": queixas,
        "diagnosticos": diagnosticos,
        "exames_alterados": exames_alterados,
        "fatores_risco": fatores_risco,
        "regras_seguranca": REGRAS_SEGURANCA_PROMPT,
    })
    return stream_com_processamento(
        stream,
        mensagem_usuario=queixas,
        fluxo="encaminhamentos",
        especialidade="multidisciplinar",
        contexto_paciente=contexto_paciente,
    )
