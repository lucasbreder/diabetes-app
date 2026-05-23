"""
Pipeline principal do Assistente Médico.
Integra LangChain (LCEL) com as bases de dados clínicas e protocolos médicos.
Usa injeção de contexto estruturado em vez de agente ReAct para maior compatibilidade.
"""

from __future__ import annotations

from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from .database import (
    buscar_ciclos_menstruais,
    buscar_exames,
    buscar_exames_atrasados,
    buscar_exames_alterados,
    buscar_medicamento,
    buscar_paciente,
    buscar_prontuarios,
    buscar_protocolos,
    buscar_triagens_vd,
    init_db,
)
from .seed_data import popular_banco

# ─────────────────────────────────────────────
# Prompt principal com contexto injetado
# ─────────────────────────────────────────────

_CHAT_PROMPT = PromptTemplate.from_template(
    """Você é uma assistente médica especializada em saúde da mulher com formação em ginecologia,
obstetrícia e saúde reprodutiva. Apoia profissionais de saúde no Brasil.

Sempre responda em português brasileiro. Seja técnica, empática e sensível ao gênero.
Quando citar protocolos, referencie FEBRASGO, INCA ou Ministério da Saúde.
Nunca tome decisões diagnósticas definitivas — você apoia, não substitui o médico.

DADOS CLÍNICOS DA PACIENTE:
{contexto_paciente}

PROTOCOLOS RELEVANTES:
{protocolos_contexto}

{chat_history}
Profissional: {input}
Assistente:"""
)


def _formatar_historico(historico: list) -> str:
    """Converte lista de HumanMessage/AIMessage em texto simples."""
    if not historico:
        return ""
    linhas = []
    for msg in historico:
        if isinstance(msg, HumanMessage):
            linhas.append(f"Profissional: {msg.content}")
        elif isinstance(msg, AIMessage):
            linhas.append(f"Assistente: {msg.content}")
    return "\n".join(linhas) + "\n"


# ─────────────────────────────────────────────
# Funções auxiliares de contexto
# ─────────────────────────────────────────────

def _montar_contexto_paciente(paciente_id: int) -> str:
    """Monta um resumo clínico completo da paciente para injeção no prompt."""
    from datetime import date

    paciente = buscar_paciente(paciente_id)
    if not paciente:
        return "Nenhuma paciente selecionada."

    hoje = date.today()
    idade = hoje.year - paciente.data_nascimento.year - (
        (hoje.month, hoje.day) < (paciente.data_nascimento.month, paciente.data_nascimento.day)
    )

    partes = [f"Nome: {paciente.nome} | Idade: {idade} anos"]

    prontuarios = buscar_prontuarios(paciente_id)
    if prontuarios:
        up = prontuarios[0]
        partes += [
            f"Histórico obstétrico: G{up.gestacoes}P{up.partos_normais + up.partos_cesareos}A{up.abortos}",
            f"Contraceptivo atual: {up.metodo_contraceptivo or 'Não informado'}",
            f"Queixas recentes: {(up.queixas or 'N/A')[:200]}",
            f"Alergias: {up.alergias or 'Nenhuma'}",
            f"Medicamentos em uso: {(up.medicamentos_uso or 'Nenhum')[:120]}",
            f"Histórico IST: {up.historico_dst or 'Nenhum relatado'}",
            f"Última consulta: {up.data_consulta} | Responsável: {up.medico_responsavel or 'N/A'}",
        ]

    ciclos = buscar_ciclos_menstruais(paciente_id, ultimos_n=3)
    if ciclos:
        resumo_ciclos = " | ".join(
            f"{c.data_inicio} (dur.{c.duracao_dias or '?'}d, dor {c.dor_escala or '?'}/10)"
            for c in ciclos
        )
        partes.append(f"Ciclos recentes: {resumo_ciclos}")

    exames = buscar_exames(paciente_id)
    if exames:
        resumo_exames = []
        for e in exames[:5]:
            status = "⚠️ALTERADO" if e.resultado_alterado else "normal"
            resumo_exames.append(f"{e.tipo_exame}({e.data_realizacao or 'N/A'},{status})")
        partes.append(f"Exames: {', '.join(resumo_exames)}")

    atrasados = buscar_exames_atrasados(paciente_id)
    if atrasados:
        partes.append(f"⚠️ EXAMES ATRASADOS: {', '.join(e.tipo_exame for e in atrasados)}")

    vd = buscar_triagens_vd(paciente_id)
    if vd:
        partes.append(f"Triagem VD mais recente: {vd[0].data_triagem} – risco {vd[0].nivel_risco.upper()}")
        if vd[0].nivel_risco in ("alto", "critico"):
            partes.append("🔴 ALERTA: Paciente com risco elevado de violência doméstica")

    return "\n".join(partes)


def _buscar_protocolos_relevantes(mensagem: str) -> str:
    """Busca protocolos relevantes para a mensagem atual."""
    from .database import buscar_protocolos
    resultados = buscar_protocolos(termo=mensagem)
    if not resultados:
        return "Nenhum protocolo específico encontrado. Use conhecimento geral de ginecologia/obstetrícia."
    # Retorna resumo dos 2 protocolos mais relevantes
    partes = []
    for p in resultados[:2]:
        resumo = p.conteudo[:600] + "..." if len(p.conteudo) > 600 else p.conteudo
        partes.append(f"[{p.titulo} – {p.fonte}]\n{resumo}")
    return "\n\n".join(partes)


# ─────────────────────────────────────────────
# Pipeline Principal
# ─────────────────────────────────────────────

class AssistenteMedico:
    """
    Assistente médica especializada em saúde da mulher.
    Usa LCEL (LangChain Expression Language) com injeção de contexto clínico.
    """

    def __init__(self, paciente_id: int | None = None, modelo: str = "llama3:latest"):
        init_db()
        popular_banco()

        self.paciente_id = paciente_id
        self.modelo = modelo
        self.historico: list = []

        self._llm = OllamaLLM(model=modelo, temperature=0.3)
        self._chain = _CHAT_PROMPT | self._llm | StrOutputParser()

    def definir_paciente(self, paciente_id: int) -> None:
        """Troca a paciente em atendimento e limpa o histórico."""
        self.paciente_id = paciente_id
        self.historico = []

    def chat(self, mensagem: str) -> str:
        """Envia uma mensagem e retorna a resposta completa."""
        contexto = (
            _montar_contexto_paciente(self.paciente_id)
            if self.paciente_id
            else "Nenhuma paciente selecionada. Respondendo com conhecimento geral em saúde da mulher."
        )
        protocolos = _buscar_protocolos_relevantes(mensagem)

        resposta = self._chain.invoke({
            "input": mensagem,
            "chat_history": _formatar_historico(self.historico),
            "contexto_paciente": contexto,
            "protocolos_contexto": protocolos,
        })

        self._atualizar_historico(mensagem, resposta)
        return resposta

    def stream_chat(self, mensagem: str) -> Iterator[str]:
        """Versão streaming do chat para uso com Streamlit."""
        contexto = (
            _montar_contexto_paciente(self.paciente_id)
            if self.paciente_id
            else "Nenhuma paciente selecionada. Respondendo com conhecimento geral em saúde da mulher."
        )
        protocolos = _buscar_protocolos_relevantes(mensagem)

        resposta_completa = ""
        for chunk in self._chain.stream({
            "input": mensagem,
            "chat_history": _formatar_historico(self.historico),
            "contexto_paciente": contexto,
            "protocolos_contexto": protocolos,
        }):
            resposta_completa += chunk
            yield chunk

        self._atualizar_historico(mensagem, resposta_completa)

    def _atualizar_historico(self, pergunta: str, resposta: str) -> None:
        self.historico.append(HumanMessage(content=pergunta))
        self.historico.append(AIMessage(content=resposta))
        # Janela deslizante de 10 turnos
        if len(self.historico) > 20:
            self.historico = self.historico[-20:]

    def limpar_historico(self) -> None:
        self.historico = []


def criar_pipeline_assistente(paciente_id: int | None = None, modelo: str = "llama3:latest") -> AssistenteMedico:
    """Factory function para criar o pipeline do assistente médico."""
    return AssistenteMedico(paciente_id=paciente_id, modelo=modelo)
