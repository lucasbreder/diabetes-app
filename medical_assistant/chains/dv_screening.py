"""
Chain de triagem de violência doméstica usando o instrumento WAST
(Woman Abuse Screening Tool) adaptado para o contexto brasileiro.

IMPORTANTE: Toda informação desta triagem deve ser tratada com máxima
confidencialidade e os dados somente devem ser acessados pelo profissional
responsável. A notificação compulsória é obrigatória conforme a Lei 11.340/2006
(Lei Maria da Penha) e as normas do SINAN.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from ..safety import REGRAS_SEGURANCA_PROMPT
from ..validation_pipeline import processar_resposta_final, stream_com_processamento

# Instrumento WAST adaptado ao português (BR) – 8 itens
PERGUNTAS_WAST = [
    "Em geral, como você descreveria seu relacionamento? (há tensão/problemas)",
    "Você e seu parceiro resolvem discussões com dificuldade?",
    "Discussões resultam em sentimentos de rebaixamento ou humilhação?",
    "Discussões terminam com ameaças contra você?",
    "Seu parceiro já lhe ameaçou ou machucou fisicamente?",
    "Você sente medo do seu parceiro?",
    "Você já foi forçada a ter relações sexuais quando não queria?",
    "Você é impedida de ver amigos/família por seu parceiro?",
]

# Pontuação: 0 = não / 1 = às vezes / 2 = sim
# Risco: <3 = baixo | 3-4 = moderado | 5-6 = alto | ≥7 = crítico

PROMPT_TRIAGEM_VD = PromptTemplate.from_template(
    """Você é uma profissional de saúde especializada no acolhimento a mulheres em situação de violência.
Abordagem empática, não-julgadora e focada na segurança da mulher.
Toda informação é estritamente confidencial (exceto notificação compulsória legal).
Nunca culpabilize a paciente. Priorize sempre a segurança da mulher e eventuais filhos.
Notificação ao SINAN é obrigatória quando confirmada violência.
Em risco crítico: acionar rede de proteção imediatamente.
Responda em português brasileiro com linguagem técnica.

{regras_seguranca}

Resultado da triagem WAST para a paciente {nome_paciente}:
Pontuação total WAST: {pontuacao}/16
Nível de risco: {nivel_risco}
Indicadores identificados: {indicadores_positivos}
Sinais físicos observados: {sinais_fisicos}
Observações clínicas: {observacoes}

Elabore o relatório confidencial com as seções: Avaliação de Risco, Indicadores Clínicos Identificados, Conduta Imediata Recomendada, Plano de Segurança, Encaminhamentos e Notificações, Recursos de Apoio (CVL 180, CREAS, Delegacia da Mulher).

RELATÓRIO CONFIDENCIAL:"""
)

PROMPT_ABORDAGEM_PACIENTE = PromptTemplate.from_template(
    """Você é uma profissional de saúde treinada para acolher mulheres em situação de vulnerabilidade.
Crie uma mensagem de acolhimento verbal, em linguagem simples e empática, sem alarmar.
Informe sobre recursos disponíveis. Responda em português brasileiro.

{regras_seguranca}

Mensagem de acolhimento para mulher com nível de risco {nivel_risco} de violência doméstica.
Inclua: acolhimento empático, recursos (180, CREAS, Delegacia da Mulher), plano de segurança básico se risco moderado/alto/crítico, reafirmação de confidencialidade.
Máximo 10 linhas.

Mensagem:"""
)


def calcular_risco_wast(pontuacoes: list[int]) -> tuple[int, str]:
    """
    Calcula o nível de risco com base nas respostas WAST.

    Args:
        pontuacoes: Lista de 8 valores (0=não, 1=às vezes, 2=sim).

    Returns:
        Tupla (pontuacao_total, nivel_risco).
    """
    total = sum(pontuacoes)
    if total < 3:
        nivel = "baixo"
    elif total < 5:
        nivel = "moderado"
    elif total < 7:
        nivel = "alto"
    else:
        nivel = "critico"
    return total, nivel


def executar_triagem_violencia(
    nome_paciente: str,
    pontuacoes_wast: list[int],
    sinais_fisicos: list[str] | None = None,
    observacoes: str = "",
    modelo: str = "llama3:latest",
) -> dict:
    """
    Executa a triagem de violência doméstica e gera orientações para o profissional.

    Args:
        nome_paciente: Nome da paciente (usado no relatório).
        pontuacoes_wast: Lista de 8 valores WAST (0=não, 1=às vezes, 2=sim).
        sinais_fisicos: Sinais físicos observados durante a consulta.
        observacoes: Observações clínicas adicionais.
        modelo: Modelo Ollama a usar.

    Returns:
        dict com 'pontuacao', 'nivel_risco', 'relatorio_profissional', 'mensagem_paciente'.
    """
    pontuacao, nivel_risco = calcular_risco_wast(pontuacoes_wast)

    indicadores_positivos = []
    for i, (pergunta, score) in enumerate(zip(PERGUNTAS_WAST, pontuacoes_wast)):
        if score > 0:
            intensidade = "às vezes" if score == 1 else "sim (confirmado)"
            indicadores_positivos.append(f"• Q{i+1}: {pergunta} → {intensidade}")

    llm = OllamaLLM(model=modelo, temperature=0.2)

    chain_relatorio = PROMPT_TRIAGEM_VD | llm | StrOutputParser()
    relatorio_bruto = chain_relatorio.invoke({
        "nome_paciente": nome_paciente,
        "pontuacao": pontuacao,
        "nivel_risco": nivel_risco.upper(),
        "indicadores_positivos": "\n".join(indicadores_positivos) if indicadores_positivos else "Nenhum indicador positivo",
        "sinais_fisicos": ", ".join(sinais_fisicos) if sinais_fisicos else "Nenhum observado",
        "observacoes": observacoes or "Sem observações adicionais",
        "regras_seguranca": REGRAS_SEGURANCA_PROMPT,
    })
    relatorio, _ = processar_resposta_final(
        relatorio_bruto,
        mensagem_usuario="violência doméstica triagem WAST",
        fluxo="vd",
        especialidade="violencia_domestica",
        nivel_risco_vd=nivel_risco,
    )

    chain_abordagem = PROMPT_ABORDAGEM_PACIENTE | llm | StrOutputParser()
    mensagem_paciente = chain_abordagem.invoke({
        "nivel_risco": nivel_risco,
        "regras_seguranca": REGRAS_SEGURANCA_PROMPT,
    })

    return {
        "pontuacao": pontuacao,
        "nivel_risco": nivel_risco,
        "relatorio_profissional": relatorio,
        "mensagem_paciente": mensagem_paciente,
        "protocolo_obrigatorio": nivel_risco in ("alto", "critico"),
        "indicadores": indicadores_positivos,
    }


def stream_triagem_violencia(
    nome_paciente: str,
    pontuacoes_wast: list[int],
    sinais_fisicos: list[str] | None = None,
    observacoes: str = "",
    modelo: str = "llama3:latest",
):
    """Versão streaming do relatório para uso com Streamlit."""
    pontuacao, nivel_risco = calcular_risco_wast(pontuacoes_wast)

    indicadores_positivos = []
    for i, (pergunta, score) in enumerate(zip(PERGUNTAS_WAST, pontuacoes_wast)):
        if score > 0:
            intensidade = "às vezes" if score == 1 else "sim (confirmado)"
            indicadores_positivos.append(f"• Q{i+1}: {pergunta} → {intensidade}")

    llm = OllamaLLM(model=modelo, temperature=0.2)
    chain = PROMPT_TRIAGEM_VD | llm | StrOutputParser()

    stream = chain.stream({
        "nome_paciente": nome_paciente,
        "pontuacao": pontuacao,
        "nivel_risco": nivel_risco.upper(),
        "indicadores_positivos": "\n".join(indicadores_positivos) if indicadores_positivos else "Nenhum indicador positivo",
        "sinais_fisicos": ", ".join(sinais_fisicos) if sinais_fisicos else "Nenhum observado",
        "observacoes": observacoes or "Sem observações adicionais",
        "regras_seguranca": REGRAS_SEGURANCA_PROMPT,
    })
    return nivel_risco, stream_com_processamento(
        stream,
        mensagem_usuario="violência doméstica triagem WAST",
        fluxo="vd",
        especialidade="violencia_domestica",
        nivel_risco_vd=nivel_risco,
    )
