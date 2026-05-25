"""
Seletor de paciente compartilhado entre as páginas de fluxo.

Permite carregar uma paciente do banco de dados (semente FEBRASGO) e pré-preencher
campos comuns dos formulários (nome, idade, histórico obstétrico, contraceptivo,
alergias, exames preventivos etc.).

A seleção é persistida em `st.session_state["paciente_id_global"]` e fica
disponível para qualquer página.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import streamlit as st


SESSION_KEY = "paciente_id_global"


@dataclass
class ContextoPaciente:
    """Dados clínicos consolidados da paciente selecionada."""
    paciente_id: int
    nome: str
    idade: int
    data_nascimento: date
    cpf: Optional[str] = None
    telefone: Optional[str] = None

    # Prontuário mais recente
    gestacoes: int = 0
    partos_normais: int = 0
    partos_cesareos: int = 0
    abortos: int = 0
    metodo_contraceptivo: Optional[str] = None
    ultima_menstruacao: Optional[date] = None
    alergias: Optional[str] = None
    medicamentos_uso: Optional[str] = None
    queixas: Optional[str] = None
    historico_dst: Optional[str] = None

    # Exames preventivos: tipo -> data_realizacao mais recente
    exames_por_tipo: dict[str, date] = field(default_factory=dict)
    tem_alterado: bool = False
    tem_atrasado: bool = False

    # Histórico de violência doméstica (mais recente)
    risco_vd: Optional[str] = None

    @property
    def paridade(self) -> int:
        return self.partos_normais + self.partos_cesareos


@st.cache_data(ttl=30, show_spinner=False)
def _carregar_contexto(paciente_id: int) -> Optional[dict[str, Any]]:
    """
    Carrega contexto da paciente como dict serializável (cacheável).
    Retorna None quando a paciente não existe.
    """
    try:
        from medical_assistant.database import (
            buscar_exames,
            buscar_exames_alterados,
            buscar_exames_atrasados,
            buscar_paciente,
            buscar_prontuarios,
            buscar_triagens_vd,
        )
    except ImportError:
        return None

    paciente = buscar_paciente(paciente_id)
    if not paciente:
        return None

    hoje = date.today()
    idade = hoje.year - paciente.data_nascimento.year - (
        (hoje.month, hoje.day) < (paciente.data_nascimento.month, paciente.data_nascimento.day)
    )

    prontuarios = buscar_prontuarios(paciente_id)
    up = prontuarios[0] if prontuarios else None

    exames_por_tipo: dict[str, date] = {}
    for e in buscar_exames(paciente_id):
        if e.data_realizacao and e.tipo_exame not in exames_por_tipo:
            exames_por_tipo[e.tipo_exame] = e.data_realizacao

    triagens = buscar_triagens_vd(paciente_id, descriptografar=False)
    risco_vd = triagens[0].nivel_risco if triagens else None

    return {
        "paciente_id": paciente.id,
        "nome": paciente.nome,
        "idade": idade,
        "data_nascimento": paciente.data_nascimento,
        "cpf": paciente.cpf,
        "telefone": paciente.telefone,
        "gestacoes": up.gestacoes if up else 0,
        "partos_normais": up.partos_normais if up else 0,
        "partos_cesareos": up.partos_cesareos if up else 0,
        "abortos": up.abortos if up else 0,
        "metodo_contraceptivo": up.metodo_contraceptivo if up else None,
        "ultima_menstruacao": up.ultima_menstruacao if up else None,
        "alergias": up.alergias if up else None,
        "medicamentos_uso": up.medicamentos_uso if up else None,
        "queixas": up.queixas if up else None,
        "historico_dst": up.historico_dst if up else None,
        "exames_por_tipo": exames_por_tipo,
        "tem_alterado": len(buscar_exames_alterados(paciente_id)) > 0,
        "tem_atrasado": len(buscar_exames_atrasados(paciente_id)) > 0,
        "risco_vd": risco_vd,
    }


def obter_contexto_paciente() -> Optional[ContextoPaciente]:
    """Retorna o contexto da paciente atualmente selecionada (ou None)."""
    pid = st.session_state.get(SESSION_KEY)
    if not pid:
        return None
    dados = _carregar_contexto(pid)
    if not dados:
        return None
    return ContextoPaciente(**dados)


def render_seletor_sidebar(
    *,
    titulo: str = "👤 Paciente em Atendimento",
    permitir_anonima: bool = True,
    on_change: Optional[callable] = None,
    mostrar_navegacao: bool = True,
) -> Optional[ContextoPaciente]:
    """
    Renderiza o seletor de paciente na sidebar e devolve o contexto carregado.

    Args:
        titulo: cabeçalho exibido.
        permitir_anonima: se True, mostra opção '— Nova/anônima —' (não vincula).
        on_change: callback chamado quando a paciente muda (recebe novo ID ou None).

    Returns:
        ContextoPaciente quando há paciente selecionada, None caso contrário.
    """
    try:
        from medical_assistant.database import init_db, listar_pacientes
        from medical_assistant.seed_data import popular_banco
        init_db()
        popular_banco()
    except ImportError as exc:
        with st.sidebar:
            st.error(f"Banco de dados clínicos indisponível: {exc}")
        return None

    with st.sidebar:
        st.markdown(f"### {titulo}")

        pacientes = listar_pacientes()
        opcoes: dict[str, Optional[int]] = {}
        if permitir_anonima:
            opcoes["— Nova / anônima —"] = None
        for p in pacientes:
            opcoes[f"{p.nome} (ID {p.id})"] = p.id

        # Recupera índice atual baseado no session_state
        atual_id = st.session_state.get(SESSION_KEY)
        labels = list(opcoes.keys())
        try:
            indice_atual = next(
                i for i, label in enumerate(labels) if opcoes[label] == atual_id
            )
        except StopIteration:
            indice_atual = 0

        selecao = st.selectbox(
            "Carregar paciente do prontuário:",
            labels,
            index=indice_atual,
            key="_paciente_selectbox",
            help="Selecione uma paciente para pré-preencher o formulário com dados do prontuário.",
        )

        novo_id = opcoes[selecao]
        if novo_id != atual_id:
            st.session_state[SESSION_KEY] = novo_id
            _carregar_contexto.clear()
            if on_change:
                on_change(novo_id)
            st.rerun()

        contexto = obter_contexto_paciente()

        if contexto:
            st.markdown(
                f"<div class='sidebar-info'>"
                f"<strong>{contexto.nome}</strong><br>"
                f"📅 {contexto.idade} anos &nbsp;|&nbsp; "
                f"G{contexto.gestacoes}P{contexto.paridade}A{contexto.abortos}<br>"
                f"💊 {contexto.metodo_contraceptivo or 'Sem contraceptivo'}<br>"
                f"{'⚠️ ' + contexto.alergias if contexto.alergias and contexto.alergias != 'Nenhuma' else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )

            badges = []
            if contexto.tem_atrasado:
                badges.append("🟠 Exames atrasados")
            if contexto.tem_alterado:
                badges.append("🔬 Resultados alterados")
            if contexto.risco_vd in ("alto", "critico"):
                badges.append(f"🔴 Risco VD: {contexto.risco_vd.upper()}")
            for b in badges:
                st.warning(b)
        else:
            st.info("Selecione uma paciente para pré-preencher os formulários, ou siga digitando manualmente.")

        if mostrar_navegacao:
            st.divider()
            st.page_link("app.py", label="🏠 Voltar ao início")
            st.page_link("pages/assistente_medico.py", label="🤖 Assistente Médico")
            st.page_link("pages/5_📜_Auditoria.py", label="📜 Auditoria")

    return contexto
