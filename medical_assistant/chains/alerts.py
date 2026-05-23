"""
Chain de alertas para exames preventivos em atraso.
Gera comunicado personalizado para a paciente e orientações ao profissional.
"""

from __future__ import annotations

from datetime import date

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from ..database import buscar_exames_atrasados, buscar_exames_alterados, buscar_paciente

PROMPT_ALERTAS = PromptTemplate.from_template(
    """Você é uma assistente de saúde preventiva especializada em saúde da mulher.
Comunique de forma clara, empática e motivadora os alertas de exames preventivos em atraso.
Use linguagem acessível para a paciente e técnica para o profissional. Seja encorajadora.
Responda sempre em português brasileiro.

Relatório de alertas preventivos:
Paciente: {nome_paciente} ({idade} anos)

Exames com prazo vencido:
{exames_atrasados}

Exames com resultados alterados:
{exames_alterados}

Data atual: {data_hoje}

Elabore: Resumo para o Profissional de Saúde, Mensagem para a Paciente (linguagem simples), Plano de Ação Sugerido, Resultados Alterados que Necessitam Seguimento.

Relatório:"""
)


def _calcular_idade(data_nascimento: date) -> int:
    hoje = date.today()
    return hoje.year - data_nascimento.year - (
        (hoje.month, hoje.day) < (data_nascimento.month, data_nascimento.day)
    )


def _formatar_exames_atrasados(exames: list) -> str:
    if not exames:
        return "Nenhum exame com prazo vencido."
    linhas = []
    for e in exames:
        dias_atraso = (date.today() - e.proximo_previsto).days
        linhas.append(
            f"• {e.tipo_exame.upper()}: vencido há {dias_atraso} dias "
            f"(data prevista: {e.proximo_previsto})"
        )
    return "\n".join(linhas)


def _formatar_exames_alterados(exames: list) -> str:
    if not exames:
        return "Nenhum resultado alterado em acompanhamento."
    linhas = []
    for e in exames:
        linhas.append(
            f"• {e.tipo_exame.upper()} ({e.data_realizacao}): "
            f"{(e.resultado or '')[:150]}..."
            f" | Próximo acompanhamento: {e.proximo_previsto or 'A definir'}"
        )
    return "\n".join(linhas)


def gerar_alertas_exames(
    paciente_id: int,
    modelo: str = "llama3:latest",
) -> dict:
    """
    Gera alertas de exames preventivos atrasados ou com resultados alterados.

    Returns:
        dict com 'tem_alertas' (bool), 'exames_atrasados' (list),
        'exames_alterados' (list) e 'texto_llm' (str gerado pela IA).
    """
    paciente = buscar_paciente(paciente_id)
    atrasados = buscar_exames_atrasados(paciente_id)
    alterados = buscar_exames_alterados(paciente_id)

    resultado = {
        "tem_alertas": bool(atrasados or alterados),
        "exames_atrasados": atrasados,
        "exames_alterados": alterados,
        "texto_llm": "",
    }

    if not resultado["tem_alertas"]:
        resultado["texto_llm"] = "✅ Todos os exames preventivos estão em dia. Parabéns pelo cuidado com a saúde!"
        return resultado

    nome = paciente.nome if paciente else "Paciente"
    idade = _calcular_idade(paciente.data_nascimento) if paciente else "N/A"

    llm = OllamaLLM(model=modelo, temperature=0.3)
    chain = PROMPT_ALERTAS | llm | StrOutputParser()

    resultado["texto_llm"] = chain.invoke({
        "nome_paciente": nome,
        "idade": idade,
        "exames_atrasados": _formatar_exames_atrasados(atrasados),
        "exames_alterados": _formatar_exames_alterados(alterados),
        "data_hoje": date.today().strftime("%d/%m/%Y"),
    })

    return resultado


def stream_alertas_exames(
    paciente_id: int,
    modelo: str = "llama3:latest",
):
    """Versão streaming para uso com Streamlit."""
    paciente = buscar_paciente(paciente_id)
    atrasados = buscar_exames_atrasados(paciente_id)
    alterados = buscar_exames_alterados(paciente_id)

    if not atrasados and not alterados:
        def _gen():
            yield "✅ Todos os exames preventivos estão em dia!"
        return _gen()

    nome = paciente.nome if paciente else "Paciente"
    idade = _calcular_idade(paciente.data_nascimento) if paciente else "N/A"

    llm = OllamaLLM(model=modelo, temperature=0.3)
    chain = PROMPT_ALERTAS | llm | StrOutputParser()

    return chain.stream({
        "nome_paciente": nome,
        "idade": idade,
        "exames_atrasados": _formatar_exames_atrasados(atrasados),
        "exames_alterados": _formatar_exames_alterados(alterados),
        "data_hoje": date.today().strftime("%d/%m/%Y"),
    })
