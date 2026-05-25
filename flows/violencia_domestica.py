"""
Fluxo de Detecção de Violência Doméstica com LangGraph.

Pipeline: Sinais de alerta → Avaliação de risco → Protocolo de segurança →
          Acionamento de equipe especializada → Documentação segura → Seguimento
"""

from datetime import datetime
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from flows.llm_utils import consultar_llm


class EstadoViolenciaDomestica(TypedDict):
    paciente_id: str
    nome_paciente: str
    idade: int
    sinais_alerta: List[str]
    relato_paciente: str
    lesoes_observadas: List[str]
    historico_atendimentos: int
    acompanhante_presente: bool
    comportamento_observado: List[str]
    avaliacao_risco: Dict[str, Any]
    protocolo_seguranca: Dict[str, Any]
    equipe_acionada: Dict[str, Any]
    documentacao: Dict[str, Any]
    plano_seguimento: Dict[str, Any]
    nivel_risco: str
    timestamp: str
    resumo_final: str


# ============================================================
# Sinais de alerta ponderados para detecção
# ============================================================

SINAIS_PESO = {
    "lesões em diferentes estágios de cicatrização": 5,
    "lesões incompatíveis com relato": 5,
    "parceiro controlador presente": 4,
    "paciente evita contato visual": 3,
    "relato inconsistente sobre lesões": 4,
    "múltiplas visitas ao pronto-socorro": 4,
    "isolamento social relatado": 3,
    "medo verbalizado": 5,
    "tentativa de minimizar lesões": 3,
    "atraso na busca por atendimento": 3,
    "lesões durante a gravidez": 5,
    "comportamento ansioso ou submisso": 3,
    "sinais de desnutrição": 3,
    "marcas de contenção": 5,
}


def identificar_sinais_alerta(state: EstadoViolenciaDomestica) -> dict:
    """Nó 1 — Identifica e classifica sinais de alerta."""
    sinais = state.get("sinais_alerta", [])
    lesoes = state.get("lesoes_observadas", [])
    comportamento = state.get("comportamento_observado", [])

    prompt = (
        f"Você é um profissional de saúde treinado em detecção de violência doméstica.\n"
        f"Paciente: {state['nome_paciente']}, {state['idade']} anos\n"
        f"Sinais observados: {', '.join(sinais)}\n"
        f"Lesões: {', '.join(lesoes)}\n"
        f"Comportamento: {', '.join(comportamento)}\n"
        f"Relato: {state.get('relato_paciente', 'Não disponível')}\n"
        f"Acompanhante presente: {'Sim' if state.get('acompanhante_presente') else 'Não'}\n"
        f"Atendimentos anteriores: {state.get('historico_atendimentos', 0)}\n\n"
        f"Analise os sinais e identifique o nível de suspeita de violência doméstica. "
        f"Responda em português brasileiro."
    )
    analise_llm = consultar_llm(prompt)

    # Pontuação baseada em regras
    score = 0
    todos_sinais = sinais + lesoes + comportamento
    for sinal in todos_sinais:
        for chave, peso in SINAIS_PESO.items():
            if chave.lower() in sinal.lower():
                score += peso

    if state.get("historico_atendimentos", 0) >= 3:
        score += 4
    if state.get("acompanhante_presente", False):
        score += 2

    if score >= 12:
        nivel = "critico"
    elif score >= 7:
        nivel = "alto"
    elif score >= 4:
        nivel = "moderado"
    else:
        nivel = "baixo"

    return {
        "avaliacao_risco": {
            "score": score,
            "sinais_identificados": todos_sinais,
            "analise_llm": analise_llm,
            "timestamp": datetime.now().isoformat(),
        },
        "nivel_risco": nivel,
    }


def avaliar_risco(state: EstadoViolenciaDomestica) -> dict:
    """Nó 2 — Avaliação detalhada de risco."""
    nivel = state.get("nivel_risco", "baixo")
    avaliacao = state.get("avaliacao_risco", {})

    prompt = (
        f"Avalie o risco de violência doméstica.\n"
        f"Score: {avaliacao.get('score', 0)}, Nível: {nivel}\n"
        f"Sinais: {', '.join(avaliacao.get('sinais_identificados', []))}\n"
        f"Gere recomendações de ação imediata. Português brasileiro."
    )
    recomendacoes = consultar_llm(prompt)
    avaliacao["recomendacoes_llm"] = recomendacoes
    avaliacao["nivel_risco"] = nivel
    return {"avaliacao_risco": avaliacao}


def definir_protocolo_seguranca(state: EstadoViolenciaDomestica) -> dict:
    """Nó 3 — Define protocolo de segurança conforme nível de risco."""
    nivel = state.get("nivel_risco", "baixo")

    protocolos = {
        "critico": {
            "acao_imediata": "Acionar SAMU + Delegacia da Mulher IMEDIATAMENTE",
            "separar_acompanhante": True,
            "ambiente_seguro": "Sala reservada com saída independente",
            "contato_emergencia": ["SAMU 192", "Ligue 180", "Delegacia da Mulher"],
            "prioridade": "MÁXIMA",
        },
        "alto": {
            "acao_imediata": "Entrevista privada obrigatória, avaliar risco iminente",
            "separar_acompanhante": True,
            "ambiente_seguro": "Consultório privado",
            "contato_emergencia": ["Ligue 180", "Delegacia da Mulher"],
            "prioridade": "ALTA",
        },
        "moderado": {
            "acao_imediata": "Entrevista individual, oferecer apoio",
            "separar_acompanhante": True,
            "ambiente_seguro": "Consultório",
            "contato_emergencia": ["Ligue 180", "CRAS"],
            "prioridade": "MODERADA",
        },
        "baixo": {
            "acao_imediata": "Manter observação e acolhimento",
            "separar_acompanhante": False,
            "ambiente_seguro": "Ambiente de atendimento padrão",
            "contato_emergencia": ["Ligue 180"],
            "prioridade": "ROTINA",
        },
    }
    return {"protocolo_seguranca": protocolos.get(nivel, protocolos["baixo"])}


def acionar_equipe(state: EstadoViolenciaDomestica) -> dict:
    """Nó 4 — Aciona equipe especializada conforme protocolo."""
    nivel = state.get("nivel_risco", "baixo")

    equipes = {
        "critico": {
            "profissionais": ["Médico responsável", "Assistente social", "Psicólogo",
                              "Segurança institucional"],
            "orgaos_externos": ["Delegacia da Mulher", "SAMU", "Ministério Público"],
            "notificacao_compulsoria": True,
            "prazo_acionamento": "IMEDIATO",
        },
        "alto": {
            "profissionais": ["Assistente social", "Psicólogo"],
            "orgaos_externos": ["Delegacia da Mulher (se paciente autorizar)"],
            "notificacao_compulsoria": True,
            "prazo_acionamento": "Até 24 horas",
        },
        "moderado": {
            "profissionais": ["Assistente social"],
            "orgaos_externos": ["CRAS"],
            "notificacao_compulsoria": False,
            "prazo_acionamento": "Até 72 horas",
        },
        "baixo": {
            "profissionais": ["Equipe de enfermagem (observação)"],
            "orgaos_externos": [],
            "notificacao_compulsoria": False,
            "prazo_acionamento": "Acompanhamento de rotina",
        },
    }
    return {"equipe_acionada": equipes.get(nivel, equipes["baixo"])}


def documentar_caso(state: EstadoViolenciaDomestica) -> dict:
    """Nó 5 — Documentação segura do caso (sigilosa)."""
    documentacao = {
        "tipo_registro": "SIGILOSO — Violência Doméstica",
        "data_registro": datetime.now().isoformat(),
        "paciente_id": state.get("paciente_id", ""),
        "nivel_risco": state.get("nivel_risco", ""),
        "sinais_documentados": state.get("avaliacao_risco", {}).get("sinais_identificados", []),
        "protocolo_aplicado": state.get("protocolo_seguranca", {}).get("prioridade", ""),
        "equipe_notificada": state.get("equipe_acionada", {}).get("profissionais", []),
        "notificacao_compulsoria": state.get("equipe_acionada", {}).get("notificacao_compulsoria", False),
        "acesso_restrito": True,
        "observacoes": "Registro protegido por sigilo. Acesso restrito à equipe autorizada.",
    }
    return {"documentacao": documentacao}


def planejar_seguimento(state: EstadoViolenciaDomestica) -> dict:
    """Nó 6 — Plano de seguimento e acompanhamento."""
    nivel = state.get("nivel_risco", "baixo")

    prompt = (
        f"Crie um plano de seguimento para caso de violência doméstica nível {nivel}.\n"
        f"Inclua: retornos, rede de apoio, recursos disponíveis. "
        f"Máximo 10 linhas. Português brasileiro."
    )
    plano_llm = consultar_llm(prompt)

    intervalos = {"critico": "24h", "alto": "48h", "moderado": "7 dias", "baixo": "30 dias"}
    plano = {
        "retorno_em": intervalos.get(nivel, "30 dias"),
        "acompanhamento_psicologico": nivel in ("critico", "alto", "moderado"),
        "acompanhamento_social": nivel in ("critico", "alto"),
        "rede_apoio": ["Ligue 180", "CRAS", "CREAS", "Casa da Mulher Brasileira"],
        "orientacoes_seguimento": plano_llm,
    }
    resumo = (
        f"=== DETECÇÃO VIOLÊNCIA DOMÉSTICA — {state['nome_paciente']} ===\n"
        f"Nível de risco: {nivel.upper()}\n"
        f"Protocolo: {state.get('protocolo_seguranca', {}).get('prioridade', '')}\n"
        f"Retorno em: {plano['retorno_em']}\n"
        f"Notificação compulsória: {state.get('equipe_acionada', {}).get('notificacao_compulsoria', False)}"
    )
    return {
        "plano_seguimento": plano,
        "resumo_final": resumo,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# Roteamento condicional
# ============================================================

def rotear_por_risco(state: EstadoViolenciaDomestica) -> str:
    """Roteamento: pula direto para acionamento de equipe se risco é crítico."""
    nivel = state.get("nivel_risco", "baixo")
    if nivel == "critico":
        return "acionar_equipe"
    return "definir_protocolo_seguranca"


# ============================================================
# Construção do Grafo
# ============================================================

def criar_fluxo_violencia_domestica() -> StateGraph:
    """Cria e compila o grafo do fluxo de detecção de violência doméstica."""
    grafo = StateGraph(EstadoViolenciaDomestica)

    grafo.add_node("identificar_sinais_alerta", identificar_sinais_alerta)
    grafo.add_node("avaliar_risco", avaliar_risco)
    grafo.add_node("definir_protocolo_seguranca", definir_protocolo_seguranca)
    grafo.add_node("acionar_equipe", acionar_equipe)
    grafo.add_node("documentar_caso", documentar_caso)
    grafo.add_node("planejar_seguimento", planejar_seguimento)

    grafo.set_entry_point("identificar_sinais_alerta")
    grafo.add_edge("identificar_sinais_alerta", "avaliar_risco")
    grafo.add_conditional_edges("avaliar_risco", rotear_por_risco, {
        "definir_protocolo_seguranca": "definir_protocolo_seguranca",
        "acionar_equipe": "acionar_equipe",
    })
    grafo.add_edge("definir_protocolo_seguranca", "acionar_equipe")
    grafo.add_edge("acionar_equipe", "documentar_caso")
    grafo.add_edge("documentar_caso", "planejar_seguimento")
    grafo.add_edge("planejar_seguimento", END)

    return grafo.compile()
