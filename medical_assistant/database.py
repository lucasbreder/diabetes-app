"""
Camada de acesso a dados: configuração do banco SQLite e operações CRUD.
"""

from __future__ import annotations

import unicodedata
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Base,
    CicloMenstrual,
    ExamePreventivo,
    Medicamento,
    Paciente,
    ProntuarioGinecologico,
    ProtocoloMedico,
    TriagemViolencia,
)
from .security_protocols import criptografar_texto_vd, descriptografar_texto_vd

# ─────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────

DB_PATH = Path("data/medical_assistant.db")
_engine = None
_SessionLocal = None


def init_db() -> None:
    """Inicializa o banco de dados e cria as tabelas se necessário."""
    global _engine, _SessionLocal
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=_engine,
        expire_on_commit=False,  # evita DetachedInstanceError após fechar sessão
    )


def _ensure_init() -> None:
    if _SessionLocal is None:
        init_db()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    _ensure_init()
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─────────────────────────────────────────────
# CRUD – Paciente
# ─────────────────────────────────────────────

def buscar_paciente(paciente_id: int) -> Optional[Paciente]:
    with get_session() as s:
        return s.get(Paciente, paciente_id)


def listar_pacientes() -> list[Paciente]:
    with get_session() as s:
        return s.query(Paciente).order_by(Paciente.nome).all()


def criar_paciente(nome: str, data_nascimento: date, cpf: str = None, telefone: str = None) -> Paciente:
    with get_session() as s:
        p = Paciente(nome=nome, data_nascimento=data_nascimento, cpf=cpf, telefone=telefone)
        s.add(p)
        s.flush()
        s.expunge(p)
        return p


# ─────────────────────────────────────────────
# CRUD – Prontuário Ginecológico
# ─────────────────────────────────────────────

def criar_prontuario(
    paciente_id: int,
    data_consulta: date,
    queixas: str = None,
    gestacoes: int = 0,
    partos_normais: int = 0,
    partos_cesareos: int = 0,
    abortos: int = 0,
    metodo_contraceptivo: str = None,
    ultima_menstruacao: date = None,
    historico_dst: str = None,
    alergias: str = None,
    medicamentos_uso: str = None,
    observacoes: str = None,
    medico_responsavel: str = None,
) -> ProntuarioGinecologico:
    """Registra um novo prontuário ginecológico para a paciente."""
    with get_session() as s:
        p = ProntuarioGinecologico(
            paciente_id=paciente_id,
            data_consulta=data_consulta,
            queixas=queixas,
            gestacoes=gestacoes,
            partos_normais=partos_normais,
            partos_cesareos=partos_cesareos,
            abortos=abortos,
            metodo_contraceptivo=metodo_contraceptivo,
            ultima_menstruacao=ultima_menstruacao,
            historico_dst=historico_dst,
            alergias=alergias,
            medicamentos_uso=medicamentos_uso,
            observacoes=observacoes,
            medico_responsavel=medico_responsavel,
        )
        s.add(p)
        s.flush()
        s.expunge(p)
        return p


def buscar_prontuarios(paciente_id: int) -> list[ProntuarioGinecologico]:
    with get_session() as s:
        return (
            s.query(ProntuarioGinecologico)
            .filter(ProntuarioGinecologico.paciente_id == paciente_id)
            .order_by(ProntuarioGinecologico.data_consulta.desc())
            .all()
        )


def buscar_ultimo_prontuario(paciente_id: int) -> Optional[ProntuarioGinecologico]:
    with get_session() as s:
        return (
            s.query(ProntuarioGinecologico)
            .filter(ProntuarioGinecologico.paciente_id == paciente_id)
            .order_by(ProntuarioGinecologico.data_consulta.desc())
            .first()
        )


# ─────────────────────────────────────────────
# CRUD – Exames Preventivos
# ─────────────────────────────────────────────

def criar_exame(
    paciente_id: int,
    tipo_exame: str,
    data_realizacao: date,
    resultado: str = None,
    resultado_alterado: bool = False,
    proximo_previsto: date = None,
    laboratorio: str = None,
    medico_solicitante: str = None,
    observacoes: str = None,
) -> ExamePreventivo:
    """Registra um novo exame preventivo para a paciente."""
    with get_session() as s:
        e = ExamePreventivo(
            paciente_id=paciente_id,
            tipo_exame=tipo_exame,
            data_realizacao=data_realizacao,
            resultado=resultado,
            resultado_alterado=resultado_alterado,
            proximo_previsto=proximo_previsto,
            laboratorio=laboratorio,
            medico_solicitante=medico_solicitante,
            observacoes=observacoes,
        )
        s.add(e)
        s.flush()
        s.expunge(e)
        return e


def buscar_exames(paciente_id: int) -> list[ExamePreventivo]:
    with get_session() as s:
        return (
            s.query(ExamePreventivo)
            .filter(ExamePreventivo.paciente_id == paciente_id)
            .order_by(ExamePreventivo.data_realizacao.desc())
            .all()
        )


def buscar_exames_atrasados(paciente_id: int) -> list[ExamePreventivo]:
    """Retorna exames com prazo vencido."""
    hoje = date.today()
    with get_session() as s:
        return (
            s.query(ExamePreventivo)
            .filter(
                ExamePreventivo.paciente_id == paciente_id,
                ExamePreventivo.proximo_previsto < hoje,
            )
            .order_by(ExamePreventivo.proximo_previsto)
            .all()
        )


def buscar_exames_alterados(paciente_id: int) -> list[ExamePreventivo]:
    """Retorna exames com resultados alterados."""
    with get_session() as s:
        return (
            s.query(ExamePreventivo)
            .filter(
                ExamePreventivo.paciente_id == paciente_id,
                ExamePreventivo.resultado_alterado.is_(True),
            )
            .order_by(ExamePreventivo.data_realizacao.desc())
            .all()
        )


# ─────────────────────────────────────────────
# CRUD – Calendário Menstrual
# ─────────────────────────────────────────────

def buscar_ciclos_menstruais(paciente_id: int, ultimos_n: int = 6) -> list[CicloMenstrual]:
    with get_session() as s:
        return (
            s.query(CicloMenstrual)
            .filter(CicloMenstrual.paciente_id == paciente_id)
            .order_by(CicloMenstrual.data_inicio.desc())
            .limit(ultimos_n)
            .all()
        )


# ─────────────────────────────────────────────
# CRUD – Triagem de Violência
# ─────────────────────────────────────────────

def buscar_triagens_vd(paciente_id: int, descriptografar: bool = True) -> list[TriagemViolencia]:
    with get_session() as s:
        triagens = (
            s.query(TriagemViolencia)
            .filter(TriagemViolencia.paciente_id == paciente_id)
            .order_by(TriagemViolencia.data_triagem.desc())
            .all()
        )
    if descriptografar:
        for t in triagens:
            if t.observacoes_confidenciais:
                t.observacoes_confidenciais = descriptografar_texto_vd(t.observacoes_confidenciais)
            if t.indicadores_identificados:
                t.indicadores_identificados = descriptografar_texto_vd(t.indicadores_identificados)
    return triagens


def registrar_triagem_vd(
    paciente_id: int,
    nivel_risco: str,
    indicadores: str,
    protocolo_acionado: bool = False,
    encaminhamentos: str = None,
    plano_seguranca: str = None,
    observacoes: str = None,
    profissional_id: str = "sessao_streamlit",
) -> TriagemViolencia:
    from .audit import registrar_log_violencia

    indicadores_cifrado = criptografar_texto_vd(indicadores) if indicadores else None
    observacoes_cifrado = criptografar_texto_vd(observacoes) if observacoes else None

    with get_session() as s:
        t = TriagemViolencia(
            paciente_id=paciente_id,
            data_triagem=date.today(),
            nivel_risco=nivel_risco,
            indicadores_identificados=indicadores_cifrado,
            protocolo_acionado=protocolo_acionado,
            encaminhamentos_realizados=encaminhamentos,
            plano_seguranca=plano_seguranca,
            observacoes_confidenciais=observacoes_cifrado,
        )
        s.add(t)
        s.flush()
        s.expunge(t)

    registrar_log_violencia(
        paciente_id=paciente_id,
        acao="escrita_triagem",
        profissional_id=profissional_id,
        detalhes={"nivel_risco": nivel_risco, "protocolo_acionado": protocolo_acionado},
    )
    if nivel_risco in ("alto", "critico"):
        from .audit import registrar_alerta_seguranca

        registrar_alerta_seguranca(
            paciente_id=paciente_id,
            nivel=nivel_risco,
            motivo=f"Triagem VD registrada com risco {nivel_risco}",
            protocolo_emergencia="VD_RISCO_ALTO" if nivel_risco == "alto" else "VD_RISCO_CRITICO",
        )
    return t


# ─────────────────────────────────────────────
# CRUD – Medicamentos
# ─────────────────────────────────────────────

def _normalizar(texto: str | None) -> str:
    """Remove acentos e converte para minúsculas, para busca insensível a Unicode."""
    if not texto:
        return ""
    decomposto = unicodedata.normalize("NFD", texto)
    sem_acento = "".join(c for c in decomposto if unicodedata.category(c) != "Mn")
    return sem_acento.casefold()


def buscar_medicamento(termo: str) -> list[Medicamento]:
    """Busca em nome comercial, princípio ativo, categoria e indicações.
    Insensível a maiúsculas/minúsculas e acentos (ex.: 'ácido fólico' == 'acido folico').
    A filtragem é feita em Python pois o ILIKE do SQLite só normaliza ASCII."""
    termo_normalizado = _normalizar(termo)
    if not termo_normalizado:
        return []
    with get_session() as s:
        candidatos = s.query(Medicamento).all()
    return [
        m for m in candidatos
        if termo_normalizado in _normalizar(m.nome_comercial)
        or termo_normalizado in _normalizar(m.principio_ativo)
        or termo_normalizado in _normalizar(m.categoria)
        or termo_normalizado in _normalizar(m.indicacoes)
    ]


# ─────────────────────────────────────────────
# CRUD – Protocolos Médicos
# ─────────────────────────────────────────────

def buscar_protocolos(termo: str = None, categoria: str = None) -> list[ProtocoloMedico]:
    """Busca protocolos por termo e/ou categoria. Insensível a acentos."""
    with get_session() as s:
        candidatos = s.query(ProtocoloMedico).all()

    categoria_norm = _normalizar(categoria) if categoria else ""
    termo_norm = _normalizar(termo) if termo else ""

    if not categoria_norm and not termo_norm:
        return candidatos

    resultado = []
    for p in candidatos:
        if categoria_norm and categoria_norm not in _normalizar(p.categoria):
            continue
        if termo_norm:
            casa = (
                termo_norm in _normalizar(p.titulo)
                or termo_norm in _normalizar(p.palavras_chave)
                or termo_norm in _normalizar(p.conteudo)
            )
            if not casa:
                continue
        resultado.append(p)
    return resultado
