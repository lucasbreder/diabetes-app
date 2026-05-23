"""
Camada de acesso a dados: configuração do banco SQLite e operações CRUD.
"""

from __future__ import annotations

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

def buscar_triagens_vd(paciente_id: int) -> list[TriagemViolencia]:
    with get_session() as s:
        return (
            s.query(TriagemViolencia)
            .filter(TriagemViolencia.paciente_id == paciente_id)
            .order_by(TriagemViolencia.data_triagem.desc())
            .all()
        )


def registrar_triagem_vd(
    paciente_id: int,
    nivel_risco: str,
    indicadores: str,
    protocolo_acionado: bool = False,
    encaminhamentos: str = None,
    plano_seguranca: str = None,
    observacoes: str = None,
) -> TriagemViolencia:
    with get_session() as s:
        t = TriagemViolencia(
            paciente_id=paciente_id,
            data_triagem=date.today(),
            nivel_risco=nivel_risco,
            indicadores_identificados=indicadores,
            protocolo_acionado=protocolo_acionado,
            encaminhamentos_realizados=encaminhamentos,
            plano_seguranca=plano_seguranca,
            observacoes_confidenciais=observacoes,
        )
        s.add(t)
        s.flush()
        s.expunge(t)
        return t


# ─────────────────────────────────────────────
# CRUD – Medicamentos
# ─────────────────────────────────────────────

def buscar_medicamento(termo: str) -> list[Medicamento]:
    """Busca por nome comercial ou princípio ativo (case-insensitive)."""
    like = f"%{termo}%"
    with get_session() as s:
        return (
            s.query(Medicamento)
            .filter(
                (Medicamento.nome_comercial.ilike(like))
                | (Medicamento.principio_ativo.ilike(like))
            )
            .all()
        )


# ─────────────────────────────────────────────
# CRUD – Protocolos Médicos
# ─────────────────────────────────────────────

def buscar_protocolos(termo: str = None, categoria: str = None) -> list[ProtocoloMedico]:
    with get_session() as s:
        q = s.query(ProtocoloMedico)
        if categoria:
            q = q.filter(ProtocoloMedico.categoria.ilike(f"%{categoria}%"))
        if termo:
            like = f"%{termo}%"
            q = q.filter(
                (ProtocoloMedico.titulo.ilike(like))
                | (ProtocoloMedico.palavras_chave.ilike(like))
                | (ProtocoloMedico.conteudo.ilike(like))
            )
        return q.all()
