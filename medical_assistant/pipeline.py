"""
Pipeline principal do Assistente Médico.
Integra LangChain (LCEL) com as bases de dados clínicas e protocolos médicos.
Usa injeção de contexto estruturado em vez de agente ReAct para maior compatibilidade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Union

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
from .safety import REGRAS_SEGURANCA_PROMPT
from .validation_pipeline import processar_resposta_final
from .seed_data import popular_banco

# ─────────────────────────────────────────────
# Prompt principal com contexto injetado
# ─────────────────────────────────────────────

_CHAT_PROMPT = PromptTemplate.from_template(
    """Você é uma assistente médica especializada em saúde da mulher com formação em ginecologia,
obstetrícia e saúde reprodutiva. Apoia profissionais de saúde no Brasil.

Sempre responda em português brasileiro. Seja técnica, empática e sensível ao gênero.
Quando citar protocolos, referencie FEBRASGO, INCA ou Ministério da Saúde.

{regras_seguranca}

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
# Eventos de progresso (para UI de streaming com etapas)
# ─────────────────────────────────────────────

TipoEtapa = Literal[
    "inicio",                # início do processamento
    "contexto_paciente",     # busca de dados clínicos no banco
    "protocolos",            # recuperação de protocolos FEBRASGO/INCA/MS
    "llm_inicio",            # primeira chamada ao LLM
    "llm_token",             # cada chunk de resposta do LLM
    "validacao",             # guardrails + auditoria + explicabilidade
    "fim",                   # conclusão (resposta_final pronta)
    "erro",                  # falha em qualquer etapa
]


@dataclass
class EventoChat:
    """Evento emitido durante o processamento de uma mensagem do chat."""
    tipo: TipoEtapa
    rotulo: str = ""           # rótulo legível para UI
    detalhe: str = ""          # contexto extra (ex.: "3 protocolos encontrados")
    payload: str = ""          # texto associado (chunk LLM, resposta final, traceback)


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

        resposta_bruta = self._chain.invoke({
            "input": mensagem,
            "chat_history": _formatar_historico(self.historico),
            "contexto_paciente": contexto,
            "protocolos_contexto": protocolos,
            "regras_seguranca": REGRAS_SEGURANCA_PROMPT,
        })
        resposta, _ = processar_resposta_final(
            resposta_bruta,
            mensagem_usuario=mensagem,
            paciente_id=self.paciente_id,
            fluxo="chat",
            especialidade="ginecologia",
            protocolos_contexto=protocolos,
            contexto_paciente=contexto,
        )

        self._atualizar_historico(mensagem, resposta)
        return resposta

    def stream_chat(self, mensagem: str) -> Iterator[str]:
        """Versão streaming do chat para uso com Streamlit (compatível com versão anterior).

        Para feedback de progresso por etapa, prefira `stream_chat_com_etapas`.
        """
        for evento in self.stream_chat_com_etapas(mensagem):
            if evento.tipo == "llm_token":
                yield evento.payload
            elif evento.tipo == "fim":
                # complemento da validação (se algo foi adicionado pelos guardrails)
                if evento.payload:
                    yield evento.payload
            elif evento.tipo == "erro":
                yield f"\n\n⚠️ {evento.rotulo}: {evento.detalhe}"

    def stream_chat_com_etapas(self, mensagem: str) -> Iterator[EventoChat]:
        """
        Versão streaming com **eventos de progresso** por etapa do pipeline.

        Emite EventoChat para: contexto_paciente → protocolos → llm_inicio →
        llm_token (vários) → validacao → fim. A UI pode usar `st.status()`
        para mostrar cada etapa em tempo real.
        """
        yield EventoChat(
            tipo="inicio",
            rotulo="Iniciando processamento",
            detalhe=f"Modelo: {self.modelo}",
        )

        # Etapa 1: contexto da paciente
        try:
            if self.paciente_id:
                contexto = _montar_contexto_paciente(self.paciente_id)
                num_linhas = len([linha for linha in contexto.split("\n") if linha.strip()])
                yield EventoChat(
                    tipo="contexto_paciente",
                    rotulo="Contexto clínico da paciente recuperado",
                    detalhe=f"{num_linhas} item(ns) consolidados do prontuário",
                )
            else:
                contexto = "Nenhuma paciente selecionada. Respondendo com conhecimento geral em saúde da mulher."
                yield EventoChat(
                    tipo="contexto_paciente",
                    rotulo="Sem paciente selecionada",
                    detalhe="Usando conhecimento geral em saúde da mulher",
                )
        except Exception as exc:
            yield EventoChat(tipo="erro", rotulo="Falha ao buscar contexto", detalhe=str(exc))
            return

        # Etapa 2: protocolos relevantes
        try:
            protocolos = _buscar_protocolos_relevantes(mensagem)
            num_protocolos = protocolos.count("[") if protocolos else 0
            if num_protocolos:
                yield EventoChat(
                    tipo="protocolos",
                    rotulo="Protocolos FEBRASGO/INCA/MS consultados",
                    detalhe=f"{num_protocolos} protocolo(s) indexado(s) recuperado(s)",
                )
            else:
                yield EventoChat(
                    tipo="protocolos",
                    rotulo="Sem protocolo indexado para esta consulta",
                    detalhe="Conhecimento geral será utilizado",
                )
        except Exception as exc:
            yield EventoChat(tipo="erro", rotulo="Falha ao buscar protocolos", detalhe=str(exc))
            return

        # Etapa 3: chamada ao LLM (streaming token a token)
        yield EventoChat(
            tipo="llm_inicio",
            rotulo=f"Consultando LLM ({self.modelo})",
            detalhe="Gerando resposta clínica…",
        )
        resposta_completa = ""
        try:
            for chunk in self._chain.stream({
                "input": mensagem,
                "chat_history": _formatar_historico(self.historico),
                "contexto_paciente": contexto,
                "protocolos_contexto": protocolos,
                "regras_seguranca": REGRAS_SEGURANCA_PROMPT,
            }):
                resposta_completa += chunk
                yield EventoChat(tipo="llm_token", payload=chunk)
        except Exception as exc:
            yield EventoChat(tipo="erro", rotulo="Falha na chamada ao LLM", detalhe=str(exc))
            return

        # Etapa 4: validação, guardrails, auditoria e explicabilidade
        yield EventoChat(
            tipo="validacao",
            rotulo="Validando resposta",
            detalhe="Guardrails de segurança, auditoria e explicabilidade",
        )
        try:
            resposta_final, _ = processar_resposta_final(
                resposta_completa,
                mensagem_usuario=mensagem,
                paciente_id=self.paciente_id,
                fluxo="chat",
                especialidade="ginecologia",
                protocolos_contexto=protocolos,
                contexto_paciente=contexto,
            )
        except Exception as exc:
            yield EventoChat(tipo="erro", rotulo="Falha na validação da resposta", detalhe=str(exc))
            return

        complemento = ""
        if len(resposta_final) > len(resposta_completa):
            complemento = resposta_final[len(resposta_completa):]

        self._atualizar_historico(mensagem, resposta_final)
        yield EventoChat(
            tipo="fim",
            rotulo="Resposta validada e registrada na auditoria",
            payload=complemento,
        )

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
