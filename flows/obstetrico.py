"""
Fluxo Obstétrico com LangGraph.

Pipeline: Dados da gestante → Avaliação de risco gestacional → Orientações →
          Agendamento de exames → Alertas de urgência → Acompanhamento contínuo
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from flows.llm_utils import consultar_llm


class EstadoObstetrico(TypedDict):
    paciente_id: str
    nome_paciente: str
    idade: int
    semanas_gestacao: int
    tipo_gestacao: str  # "única" ou "gemelar"
    gestacoes_anteriores: int
    partos_anteriores: int
    abortos_anteriores: int
    comorbidades: List[str]
    medicamentos_em_uso: List[str]
    pressao_arterial: str
    peso_atual: float
    altura: float
    glicemia_jejum: float
    grupo_sanguineo: str
    queixas_atuais: List[str]
    avaliacao_risco: Dict[str, Any]
    orientacoes: str
    exames_agendados: List[Dict[str, str]]
    alertas_urgencia: List[str]
    plano_acompanhamento: Dict[str, Any]
    nivel_risco: str
    timestamp: str
    resumo_final: str


def _resumo_paciente_obstetrico(state: EstadoObstetrico) -> str:
    altura = state.get("altura", 1.60)
    peso = state.get("peso_atual", 65.0)
    imc = peso / (altura ** 2) if altura > 0 else 0
    return (
        f"Gestante: {state.get('nome_paciente', 'N/I')} | "
        f"Idade: {state.get('idade', 'N/I')} | IG: {state.get('semanas_gestacao', 'N/I')} semanas | "
        f"IMC: {imc:.1f}\n"
        f"Gestação: {state.get('tipo_gestacao', 'única')} | "
        f"G{state.get('gestacoes_anteriores', 0)}P{state.get('partos_anteriores', 0)}"
        f"A{state.get('abortos_anteriores', 0)}\n"
        f"Comorbidades: {', '.join(state.get('comorbidades', [])) or 'Nenhuma'}\n"
        f"Medicamentos: {', '.join(state.get('medicamentos_em_uso', [])) or 'Nenhum'}\n"
        f"PA: {state.get('pressao_arterial', 'N/A')} | Glicemia jejum: {state.get('glicemia_jejum', 'N/A')}\n"
        f"Queixas atuais: {', '.join(state.get('queixas_atuais', [])) or 'Nenhuma'}"
    )


def coletar_dados_gestante(state: EstadoObstetrico) -> dict:
    """Nó 1 — Consolida e valida dados da gestante."""
    altura = state.get("altura", 1.60)
    peso = state.get("peso_atual", 65.0)
    imc = peso / (altura ** 2) if altura > 0 else 0

    dados_consolidados = {
        "idade": state["idade"],
        "imc": round(imc, 1),
        "semanas": state["semanas_gestacao"],
        "trimestre": (
            "1º trimestre" if state["semanas_gestacao"] <= 13
            else "2º trimestre" if state["semanas_gestacao"] <= 27
            else "3º trimestre"
        ),
        "gestacoes": state.get("gestacoes_anteriores", 0),
        "paridade": state.get("partos_anteriores", 0),
        "abortos": state.get("abortos_anteriores", 0),
        "comorbidades": state.get("comorbidades", []),
        "glicemia": state.get("glicemia_jejum", 0),
        "pa": state.get("pressao_arterial", "N/A"),
    }
    return {"avaliacao_risco": {"dados_consolidados": dados_consolidados}}


def avaliar_risco_gestacional(state: EstadoObstetrico) -> dict:
    """Nó 2 — Avalia risco gestacional por regras + LLM."""
    dados = state.get("avaliacao_risco", {}).get("dados_consolidados", {})
    comorbidades = state.get("comorbidades", [])
    queixas = ", ".join(state.get("queixas_atuais", [])) or "Nenhuma"

    # Avaliação por regras
    fatores_alto = []
    fatores_moderado = []

    if state["idade"] < 16 or state["idade"] > 40:
        fatores_alto.append(f"Idade materna: {state['idade']} anos")
    if state.get("abortos_anteriores", 0) >= 2:
        fatores_moderado.append("Histórico de abortos recorrentes")
    if state.get("tipo_gestacao") == "gemelar":
        fatores_alto.append("Gestação gemelar")
    if state.get("glicemia_jejum", 0) > 92:
        fatores_alto.append(f"Glicemia de jejum elevada: {state.get('glicemia_jejum')} mg/dL")

    alto_risco_comorb = ["diabetes", "hipertensão", "pré-eclâmpsia", "HIV",
                         "cardiopatia", "epilepsia", "doença renal"]
    for c in comorbidades:
        if any(ar in c.lower() for ar in alto_risco_comorb):
            fatores_alto.append(f"Comorbidade: {c}")

    if fatores_alto:
        nivel = "alto"
    elif fatores_moderado:
        nivel = "moderado"
    else:
        nivel = "habitual"

    resumo = _resumo_paciente_obstetrico(state)
    prompt = (
        f"Avalie o risco gestacional.\n{resumo}\n"
        f"Forneça análise de risco e recomendações. Português brasileiro."
    )
    analise_llm = consultar_llm(
        prompt,
        fluxo="obstetrico",
        especialidade="obstetricia",
        contexto_paciente=resumo,
        incluir_explicabilidade=False,
    )

    avaliacao = state.get("avaliacao_risco", {})
    avaliacao.update({
        "fatores_alto_risco": fatores_alto,
        "fatores_moderado_risco": fatores_moderado,
        "nivel_risco": nivel,
        "analise_llm": analise_llm,
    })
    return {"avaliacao_risco": avaliacao, "nivel_risco": nivel}


def gerar_orientacoes_obstetricas(state: EstadoObstetrico) -> dict:
    """Nó 3 — Orientações específicas por trimestre e risco."""
    semanas = state["semanas_gestacao"]
    nivel = state.get("nivel_risco", "habitual")

    resumo = _resumo_paciente_obstetrico(state)
    prompt = (
        f"Gere orientações obstétricas para gestante de {semanas} semanas, "
        f"risco {nivel}.\n{resumo}\n"
        f"Inclua: alimentação, atividade física, sinais de alerta, medicações. "
        f"Português brasileiro. Máximo 15 linhas."
    )
    return {"orientacoes": consultar_llm(
        prompt,
        fluxo="obstetrico",
        especialidade="obstetricia",
        contexto_paciente=resumo,
        incluir_explicabilidade=True,
    )}


def agendar_exames_obstetricos(state: EstadoObstetrico) -> dict:
    """Nó 4 — Agenda exames conforme trimestre e risco."""
    semanas = state["semanas_gestacao"]
    nivel = state.get("nivel_risco", "habitual")

    # Exames por trimestre (protocolo SUS)
    exames_1tri = [
        {"nome": "Hemograma", "prazo": "Imediato", "prioridade": "rotina"},
        {"nome": "Tipagem sanguínea + Coombs", "prazo": "Imediato", "prioridade": "rotina"},
        {"nome": "Glicemia de jejum", "prazo": "Imediato", "prioridade": "rotina"},
        {"nome": "Sorologia (HIV, Sífilis, Hepatite B)", "prazo": "Imediato", "prioridade": "rotina"},
        {"nome": "Urina tipo I + urocultura", "prazo": "Imediato", "prioridade": "rotina"},
        {"nome": "USG obstétrica (datação)", "prazo": "Até 13 semanas", "prioridade": "rotina"},
    ]
    exames_2tri = [
        {"nome": "TOTG 75g", "prazo": "24-28 semanas", "prioridade": "rotina"},
        {"nome": "USG morfológica", "prazo": "20-24 semanas", "prioridade": "rotina"},
        {"nome": "Hemograma de controle", "prazo": "Imediato", "prioridade": "rotina"},
    ]
    exames_3tri = [
        {"nome": "Estreptococo grupo B", "prazo": "35-37 semanas", "prioridade": "rotina"},
        {"nome": "Sorologia (repetição)", "prazo": "Imediato", "prioridade": "rotina"},
        {"nome": "USG obstétrica (crescimento)", "prazo": "Imediato", "prioridade": "rotina"},
        {"nome": "Cardiotocografia", "prazo": "A partir de 36 sem", "prioridade": "rotina"},
    ]

    if semanas <= 13:
        exames = exames_1tri
    elif semanas <= 27:
        exames = exames_2tri
    else:
        exames = exames_3tri

    if nivel == "alto":
        exames.append({"nome": "Doppler de artérias uterinas", "prazo": "Imediato", "prioridade": "urgente"})
        exames.append({"nome": "Ecocardiograma fetal", "prazo": "24-28 semanas", "prioridade": "urgente"})

    return {"exames_agendados": exames}


def verificar_alertas_urgencia(state: EstadoObstetrico) -> dict:
    """Nó 5 — Verifica condições que exigem alerta imediato."""
    alertas = []
    queixas = [q.lower() for q in state.get("queixas_atuais", [])]

    emergencias = {
        "sangramento vaginal": "⚠️ SANGRAMENTO — Avaliação obstétrica IMEDIATA",
        "perda de líquido": "⚠️ POSSÍVEL RPMO — Avaliar amniorrexe",
        "contrações regulares": "⚠️ POSSÍVEL TRABALHO DE PARTO",
        "ausência de movimentos fetais": "🚨 EMERGÊNCIA — Avaliar vitalidade fetal",
        "cefaleia intensa": "⚠️ POSSÍVEL PRÉ-ECLÂMPSIA",
        "visão turva": "🚨 EMERGÊNCIA — Suspeita de eclâmpsia iminente",
        "edema súbito": "⚠️ POSSÍVEL PRÉ-ECLÂMPSIA",
        "dor abdominal intensa": "🚨 EMERGÊNCIA — Avaliar DPP/rotura",
        "febre": "⚠️ INFECÇÃO — Investigar foco",
    }

    for queixa in queixas:
        for chave, alerta in emergencias.items():
            if chave in queixa:
                alertas.append(alerta)

    if state.get("glicemia_jejum", 0) > 126:
        alertas.append("⚠️ DIABETES GESTACIONAL NÃO CONTROLADO")

    vistos = set()
    deduplicados = [a for a in alertas if not (a in vistos or vistos.add(a))]
    return {"alertas_urgencia": deduplicados}


def planejar_acompanhamento(state: EstadoObstetrico) -> dict:
    """Nó 6 — Plano de acompanhamento contínuo."""
    nivel = state.get("nivel_risco", "habitual")
    semanas = state["semanas_gestacao"]
    alertas = state.get("alertas_urgencia", [])

    # Frequência de consultas conforme protocolo
    if nivel == "alto" or alertas:
        freq = "Semanal" if semanas >= 36 else "Quinzenal"
    elif nivel == "moderado":
        freq = "Quinzenal" if semanas >= 28 else "Mensal"
    else:
        freq = "Mensal" if semanas < 28 else ("Quinzenal" if semanas < 36 else "Semanal")

    proxima = datetime.now() + (
        timedelta(days=7) if freq == "Semanal"
        else timedelta(days=14) if freq == "Quinzenal"
        else timedelta(days=30)
    )

    plano = {
        "frequencia_consultas": freq,
        "proxima_consulta": proxima.strftime("%d/%m/%Y"),
        "classificacao_risco": nivel,
        "alertas_ativos": len(alertas),
        "encaminhamento_alto_risco": nivel == "alto",
        "exames_pendentes": len(state.get("exames_agendados", [])),
    }

    resumo = (
        f"=== ACOMPANHAMENTO OBSTÉTRICO — {state['nome_paciente']} ===\n"
        f"IG: {semanas} semanas | Risco: {nivel.upper()}\n"
        f"Próxima consulta: {plano['proxima_consulta']} ({freq})\n"
        f"Alertas ativos: {len(alertas)}\n"
        f"Exames agendados: {plano['exames_pendentes']}"
    )
    return {
        "plano_acompanhamento": plano,
        "resumo_final": resumo,
        "timestamp": datetime.now().isoformat(),
    }


def criar_fluxo_obstetrico() -> StateGraph:
    """Cria e compila o grafo do fluxo obstétrico."""
    grafo = StateGraph(EstadoObstetrico)

    grafo.add_node("coletar_dados_gestante", coletar_dados_gestante)
    grafo.add_node("avaliar_risco_gestacional", avaliar_risco_gestacional)
    grafo.add_node("gerar_orientacoes", gerar_orientacoes_obstetricas)
    grafo.add_node("agendar_exames", agendar_exames_obstetricos)
    grafo.add_node("verificar_alertas", verificar_alertas_urgencia)
    grafo.add_node("planejar_acompanhamento", planejar_acompanhamento)

    grafo.set_entry_point("coletar_dados_gestante")
    grafo.add_edge("coletar_dados_gestante", "avaliar_risco_gestacional")
    grafo.add_edge("avaliar_risco_gestacional", "gerar_orientacoes")
    grafo.add_edge("gerar_orientacoes", "agendar_exames")
    grafo.add_edge("agendar_exames", "verificar_alertas")
    grafo.add_edge("verificar_alertas", "planejar_acompanhamento")
    grafo.add_edge("planejar_acompanhamento", END)

    return grafo.compile()
