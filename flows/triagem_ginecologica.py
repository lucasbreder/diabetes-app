"""
Fluxo de Triagem Ginecológica com LangGraph.

Pipeline: Sintomas relatados → Análise de risco → Classificação de urgência →
          Sugestão de exames → Orientações iniciais → Agendamento apropriado
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from flows.llm_utils import consultar_llm


class EstadoTriagemGinecologica(TypedDict):
    paciente_id: str
    nome_paciente: str
    idade: int
    sintomas: List[str]
    historico_menstrual: str
    uso_contraceptivo: str
    gestacoes_anteriores: int
    ultima_consulta_gineco: str
    historico_familiar: List[str]
    queixas_adicionais: str
    analise_risco: Dict[str, Any]
    classificacao_urgencia: Dict[str, Any]
    exames_sugeridos: List[Dict[str, str]]
    orientacoes: str
    agendamento: Dict[str, Any]
    timestamp: str
    resumo_final: str


def analisar_sintomas(state: EstadoTriagemGinecologica) -> dict:
    """Nó 1 — Analisa sintomas e identifica padrões de risco via LLM."""
    sintomas_texto = ", ".join(state["sintomas"])
    hist_fam = ", ".join(state.get("historico_familiar", [])) or "Nenhum"

    prompt = (
        f"Você é um sistema de triagem ginecológica.\n"
        f"Paciente: {state['nome_paciente']}, {state['idade']} anos\n"
        f"Sintomas: {sintomas_texto}\n"
        f"Histórico menstrual: {state['historico_menstrual']}\n"
        f"Contraceptivo: {state['uso_contraceptivo']}\n"
        f"Gestações: {state['gestacoes_anteriores']}\n"
        f"Última consulta: {state['ultima_consulta_gineco']}\n"
        f"Histórico familiar: {hist_fam}\n\n"
        f"Analise os riscos e liste condições suspeitas, fatores de risco e "
        f"nível de risco (baixo/moderado/alto). Responda em português."
    )
    resposta = consultar_llm(prompt)

    # Fallback baseado em regras
    alto = ["sangramento intenso", "dor abdominal aguda", "massa palpável", "febre alta"]
    moderado = ["corrimento anormal", "dor pélvica", "irregularidade menstrual"]
    nivel = "baixo"
    for s in [x.lower() for x in state["sintomas"]]:
        if any(a in s for a in alto):
            nivel = "alto"
            break
        if any(m in s for m in moderado):
            nivel = "moderado"

    return {"analise_risco": {
        "sintomas_relatados": state["sintomas"],
        "nivel_risco_regras": nivel,
        "analise_llm": resposta,
        "timestamp": datetime.now().isoformat(),
    }}


def classificar_urgencia(state: EstadoTriagemGinecologica) -> dict:
    """Nó 2 — Classifica urgência com base na análise de risco."""
    nivel = state.get("analise_risco", {}).get("nivel_risco_regras", "baixo")
    mapa = {
        "alto": {"codigo": "VERMELHO", "descricao": "Atendimento imediato / Emergência",
                 "tempo_maximo": "Imediato", "encaminhamento": "Pronto-socorro ginecológico"},
        "moderado": {"codigo": "AMARELO", "descricao": "Atendimento prioritário",
                     "tempo_maximo": "Até 48h", "encaminhamento": "Consulta prioritária"},
        "baixo": {"codigo": "VERDE", "descricao": "Atendimento eletivo",
                  "tempo_maximo": "Até 15 dias", "encaminhamento": "Consulta de rotina"},
    }
    classificacao = mapa.get(nivel, mapa["baixo"])
    classificacao["nivel_risco_origem"] = nivel
    return {"classificacao_urgencia": classificacao}


def sugerir_exames(state: EstadoTriagemGinecologica) -> dict:
    """Nó 3 — Sugere exames laboratoriais e de imagem."""
    urgencia = state.get("classificacao_urgencia", {})
    prompt = (
        f"Sugira exames ginecológicos para: {state['nome_paciente']}, "
        f"{state['idade']} anos. Sintomas: {', '.join(state['sintomas'])}. "
        f"Urgência: {urgencia.get('codigo', 'N/A')}. "
        f"Liste até 6 exames com justificativa. Português brasileiro."
    )
    resposta_llm = consultar_llm(prompt)

    exames = [
        {"nome": "Papanicolau", "justificativa": "Rastreamento cervical", "prioridade": "rotina"},
        {"nome": "USG transvaginal", "justificativa": "Avaliação uterina/ovariana", "prioridade": "rotina"},
        {"nome": "Hemograma completo", "justificativa": "Investigação de anemia", "prioridade": "rotina"},
    ]
    if urgencia.get("codigo") == "VERMELHO":
        exames.insert(0, {"nome": "Beta-HCG", "justificativa": "Exclusão de gravidez", "prioridade": "urgente"})
    return {"exames_sugeridos": exames, "orientacoes": resposta_llm}


def gerar_orientacoes(state: EstadoTriagemGinecologica) -> dict:
    """Nó 4 — Orientações iniciais personalizadas para a paciente."""
    urgencia = state.get("classificacao_urgencia", {})
    exames_nomes = ", ".join(e["nome"] for e in state.get("exames_sugeridos", []))
    prompt = (
        f"Gere orientações para a paciente {state['nome_paciente']}, {state['idade']} anos.\n"
        f"Urgência: {urgencia.get('codigo', '')}. Exames: {exames_nomes}.\n"
        f"Sintomas: {', '.join(state['sintomas'])}.\n"
        f"Inclua: preparo para exames, sinais de alerta, cuidados gerais. "
        f"Seja empática. Português brasileiro. Máximo 15 linhas."
    )
    return {"orientacoes": consultar_llm(prompt)}


def realizar_agendamento(state: EstadoTriagemGinecologica) -> dict:
    """Nó 5 — Agendamento com base na classificação de urgência."""
    urgencia = state.get("classificacao_urgencia", {})
    codigo = urgencia.get("codigo", "VERDE")
    prazos = {"VERMELHO": timedelta(hours=0), "AMARELO": timedelta(days=2), "VERDE": timedelta(days=15)}
    data = datetime.now() + prazos.get(codigo, timedelta(days=15))
    espec = {"VERMELHO": "Ginecologista de plantão", "AMARELO": "Consulta prioritária", "VERDE": "Consulta eletiva"}

    agendamento = {
        "data_sugerida": data.strftime("%d/%m/%Y %H:%M"),
        "especialidade": espec.get(codigo, "Ginecologista"),
        "tipo_consulta": urgencia.get("descricao", "Rotina"),
        "observacoes": f"Classificação: {codigo}",
    }
    resumo = (
        f"=== TRIAGEM GINECOLÓGICA — {state['nome_paciente']} ===\n"
        f"Classificação: {codigo} ({urgencia.get('descricao', '')})\n"
        f"Agendamento: {agendamento['data_sugerida']}\n"
        f"Exames: {len(state.get('exames_sugeridos', []))}"
    )
    return {"agendamento": agendamento, "resumo_final": resumo, "timestamp": datetime.now().isoformat()}


def criar_fluxo_triagem_ginecologica() -> StateGraph:
    """Cria e compila o grafo do fluxo de triagem ginecológica."""
    grafo = StateGraph(EstadoTriagemGinecologica)
    grafo.add_node("analisar_sintomas", analisar_sintomas)
    grafo.add_node("classificar_urgencia", classificar_urgencia)
    grafo.add_node("sugerir_exames", sugerir_exames)
    grafo.add_node("gerar_orientacoes", gerar_orientacoes)
    grafo.add_node("realizar_agendamento", realizar_agendamento)

    grafo.set_entry_point("analisar_sintomas")
    grafo.add_edge("analisar_sintomas", "classificar_urgencia")
    grafo.add_edge("classificar_urgencia", "sugerir_exames")
    grafo.add_edge("sugerir_exames", "gerar_orientacoes")
    grafo.add_edge("gerar_orientacoes", "realizar_agendamento")
    grafo.add_edge("realizar_agendamento", END)
    return grafo.compile()
