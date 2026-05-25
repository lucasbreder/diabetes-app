"""
Fluxo de Prevenção com LangGraph.

Pipeline: Histórico da paciente → Identificação de exames devidos →
          Orientações preventivas → Agendamento automático → Lembretes personalizados
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from flows.llm_utils import consultar_llm


class EstadoPrevencao(TypedDict):
    paciente_id: str
    nome_paciente: str
    idade: int
    sexo: str
    historico_pessoal: List[str]
    historico_familiar: List[str]
    ultimo_papanicolau: str  # data ISO ou "nunca"
    ultima_mamografia: str
    ultima_densitometria: str
    ultimo_check_up: str
    vacinas_em_dia: List[str]
    habitos: Dict[str, Any]  # tabagismo, etilismo, atividade_fisica, etc.
    comorbidades: List[str]
    exames_devidos: List[Dict[str, Any]]
    orientacoes_preventivas: str
    agendamentos: List[Dict[str, str]]
    lembretes: List[Dict[str, str]]
    timestamp: str
    resumo_final: str


# ============================================================
# Protocolos de rastreamento por faixa etária (baseado em diretrizes brasileiras)
# ============================================================

PROTOCOLOS_RASTREAMENTO = {
    "papanicolau": {
        "idade_inicio": 25,
        "idade_fim": 64,
        "intervalo_anos": 3,
        "descricao": "Citologia oncótica cervical",
        "observacao": "Após 2 exames normais anuais consecutivos, trienal",
    },
    "mamografia": {
        "idade_inicio": 50,
        "idade_fim": 69,
        "intervalo_anos": 2,
        "descricao": "Mamografia bilateral",
        "observacao": "Antecipar para 40 anos se histórico familiar de 1º grau",
    },
    "densitometria": {
        "idade_inicio": 65,
        "idade_fim": 999,
        "intervalo_anos": 2,
        "descricao": "Densitometria óssea",
        "observacao": "Antecipar se menopausa precoce ou uso de corticoides",
    },
    "colesterol": {
        "idade_inicio": 20,
        "idade_fim": 999,
        "intervalo_anos": 5,
        "descricao": "Perfil lipídico",
        "observacao": "Anual se fatores de risco cardiovascular",
    },
    "glicemia": {
        "idade_inicio": 45,
        "idade_fim": 999,
        "intervalo_anos": 3,
        "descricao": "Glicemia de jejum",
        "observacao": "Antecipar se IMC > 25 ou histórico familiar de diabetes",
    },
    "colonoscopia": {
        "idade_inicio": 50,
        "idade_fim": 75,
        "intervalo_anos": 10,
        "descricao": "Colonoscopia de rastreamento",
        "observacao": "Antecipar se histórico familiar de câncer colorretal",
    },
}


def analisar_historico(state: EstadoPrevencao) -> dict:
    """Nó 1 — Analisa histórico e identifica fatores de risco preventivos."""
    prompt = (
        f"Analise o perfil preventivo da paciente.\n"
        f"Paciente: {state['nome_paciente']}, {state['idade']} anos\n"
        f"Histórico pessoal: {', '.join(state.get('historico_pessoal', [])) or 'Nenhum'}\n"
        f"Histórico familiar: {', '.join(state.get('historico_familiar', [])) or 'Nenhum'}\n"
        f"Comorbidades: {', '.join(state.get('comorbidades', [])) or 'Nenhuma'}\n"
        f"Hábitos: {state.get('habitos', {})}\n"
        f"Identifique fatores de risco e rastreamentos prioritários. Português brasileiro."
    )
    analise = consultar_llm(prompt)
    # Guardamos a análise nas orientações temporariamente
    return {"orientacoes_preventivas": analise}


def identificar_exames_devidos(state: EstadoPrevencao) -> dict:
    """Nó 2 — Identifica exames de rastreamento devidos conforme protocolos."""
    idade = state["idade"]
    hist_familiar = [h.lower() for h in state.get("historico_familiar", [])]
    exames_devidos = []

    def _exame_atrasado(ultimo: str, intervalo_anos: int) -> bool:
        if not ultimo or ultimo.lower() in ("nunca", "n/a", ""):
            return True
        try:
            dt = datetime.fromisoformat(ultimo)
            return (datetime.now() - dt).days > intervalo_anos * 365
        except ValueError:
            return True

    for nome, proto in PROTOCOLOS_RASTREAMENTO.items():
        idade_inicio = proto["idade_inicio"]

        # Antecipação por histórico familiar
        if nome == "mamografia" and any("câncer de mama" in h for h in hist_familiar):
            idade_inicio = 40
        if nome == "colonoscopia" and any("câncer" in h and "colon" in h for h in hist_familiar):
            idade_inicio = 40
        if nome == "glicemia" and any("diabetes" in h for h in hist_familiar):
            idade_inicio = 30

        if idade < idade_inicio or idade > proto["idade_fim"]:
            continue

        # Verificar se está em dia
        mapa_datas = {
            "papanicolau": state.get("ultimo_papanicolau", ""),
            "mamografia": state.get("ultima_mamografia", ""),
            "densitometria": state.get("ultima_densitometria", ""),
        }
        ultimo = mapa_datas.get(nome, state.get("ultimo_check_up", ""))

        if _exame_atrasado(ultimo, proto["intervalo_anos"]):
            exames_devidos.append({
                "exame": proto["descricao"],
                "protocolo": nome,
                "intervalo": f"A cada {proto['intervalo_anos']} ano(s)",
                "observacao": proto["observacao"],
                "status": "ATRASADO" if ultimo and ultimo.lower() != "nunca" else "NUNCA REALIZADO",
            })

    return {"exames_devidos": exames_devidos}


def gerar_orientacoes_preventivas(state: EstadoPrevencao) -> dict:
    """Nó 3 — Orientações preventivas personalizadas via LLM."""
    exames = state.get("exames_devidos", [])
    exames_txt = "; ".join(e["exame"] for e in exames) or "Nenhum exame pendente"
    habitos = state.get("habitos", {})

    prompt = (
        f"Gere orientações preventivas personalizadas.\n"
        f"Paciente: {state['nome_paciente']}, {state['idade']} anos\n"
        f"Exames pendentes: {exames_txt}\n"
        f"Comorbidades: {', '.join(state.get('comorbidades', []))}\n"
        f"Tabagismo: {habitos.get('tabagismo', 'N/A')}\n"
        f"Atividade física: {habitos.get('atividade_fisica', 'N/A')}\n"
        f"Alimentação: {habitos.get('alimentacao', 'N/A')}\n\n"
        f"Inclua: orientações sobre rastreamento, hábitos saudáveis, vacinação. "
        f"Seja empática e acolhedora. Português brasileiro. Máximo 15 linhas."
    )
    return {"orientacoes_preventivas": consultar_llm(prompt)}


def agendar_automaticamente(state: EstadoPrevencao) -> dict:
    """Nó 4 — Cria agendamentos automáticos para exames devidos."""
    exames = state.get("exames_devidos", [])
    agendamentos = []

    base = datetime.now() + timedelta(days=7)
    for i, exame in enumerate(exames):
        data = base + timedelta(days=i * 7)
        prioridade = "ALTA" if exame.get("status") == "NUNCA REALIZADO" else "NORMAL"
        agendamentos.append({
            "exame": exame["exame"],
            "data_sugerida": data.strftime("%d/%m/%Y"),
            "prioridade": prioridade,
            "preparo": _obter_preparo(exame.get("protocolo", "")),
            "status": "agendado",
        })

    return {"agendamentos": agendamentos}


def _obter_preparo(protocolo: str) -> str:
    """Retorna instruções de preparo para cada tipo de exame."""
    preparos = {
        "papanicolau": "Evitar relações sexuais 48h antes. Não usar duchas vaginais.",
        "mamografia": "Não usar desodorante/talco no dia. Levar exames anteriores.",
        "densitometria": "Não requer preparo específico.",
        "colesterol": "Jejum de 12 horas.",
        "glicemia": "Jejum de 8 horas.",
        "colonoscopia": "Preparo intestinal conforme orientação médica.",
    }
    return preparos.get(protocolo, "Consultar orientações do laboratório.")


def gerar_lembretes(state: EstadoPrevencao) -> dict:
    """Nó 5 — Gera lembretes personalizados para cada agendamento."""
    agendamentos = state.get("agendamentos", [])
    exames_devidos = state.get("exames_devidos", [])
    lembretes = []

    for ag in agendamentos:
        lembretes.append({
            "tipo": "agendamento",
            "mensagem": f"📅 Lembrete: {ag['exame']} agendado para {ag['data_sugerida']}. {ag.get('preparo', '')}",
            "data_envio": (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y"),
            "canal": "SMS/WhatsApp",
        })

    # Lembretes de rotina futura
    for exame in exames_devidos:
        proto = PROTOCOLOS_RASTREAMENTO.get(exame.get("protocolo", ""), {})
        if proto:
            intervalo = proto.get("intervalo_anos", 1)
            prox = datetime.now() + timedelta(days=intervalo * 365)
            lembretes.append({
                "tipo": "preventivo",
                "mensagem": f"🔔 Próximo {exame['exame']}: {prox.strftime('%m/%Y')}",
                "data_envio": (prox - timedelta(days=30)).strftime("%d/%m/%Y"),
                "canal": "SMS/WhatsApp",
            })

    resumo = (
        f"=== PREVENÇÃO — {state['nome_paciente']} ===\n"
        f"Exames devidos: {len(exames_devidos)}\n"
        f"Agendamentos criados: {len(agendamentos)}\n"
        f"Lembretes programados: {len(lembretes)}"
    )

    return {
        "lembretes": lembretes,
        "resumo_final": resumo,
        "timestamp": datetime.now().isoformat(),
    }


def criar_fluxo_prevencao() -> StateGraph:
    """Cria e compila o grafo do fluxo de prevenção."""
    grafo = StateGraph(EstadoPrevencao)

    grafo.add_node("analisar_historico", analisar_historico)
    grafo.add_node("identificar_exames_devidos", identificar_exames_devidos)
    grafo.add_node("gerar_orientacoes_preventivas", gerar_orientacoes_preventivas)
    grafo.add_node("agendar_automaticamente", agendar_automaticamente)
    grafo.add_node("gerar_lembretes", gerar_lembretes)

    grafo.set_entry_point("analisar_historico")
    grafo.add_edge("analisar_historico", "identificar_exames_devidos")
    grafo.add_edge("identificar_exames_devidos", "gerar_orientacoes_preventivas")
    grafo.add_edge("gerar_orientacoes_preventivas", "agendar_automaticamente")
    grafo.add_edge("agendar_automaticamente", "gerar_lembretes")
    grafo.add_edge("gerar_lembretes", END)

    return grafo.compile()
