"""
Modelos SQLAlchemy (ORM) e Pydantic (validação) para o assistente médico.
"""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Base SQLAlchemy
# ─────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────
# Enumerações
# ─────────────────────────────────────────────

class NivelRiscoVD(str, enum.Enum):
    BAIXO = "baixo"
    MODERADO = "moderado"
    ALTO = "alto"
    CRITICO = "critico"


class TipoExame(str, enum.Enum):
    PAPANICOLAU = "papanicolau"
    MAMOGRAFIA = "mamografia"
    ULTRASSOM_PELVICO = "ultrassom_pelvico"
    DENSITOMETRIA = "densitometria"
    COLPOSCOPIA = "colposcopia"
    HPV_TESTE = "hpv_teste"
    CA125 = "ca125"
    OUTRO = "outro"


# ─────────────────────────────────────────────
# Tabelas ORM
# ─────────────────────────────────────────────

class Paciente(Base):
    __tablename__ = "pacientes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nome = Column(String(200), nullable=False)
    data_nascimento = Column(Date, nullable=False)
    cpf = Column(String(14), unique=True, index=True)
    telefone = Column(String(20))
    criado_em = Column(DateTime, default=datetime.utcnow)

    prontuarios = relationship(
        "ProntuarioGinecologico", back_populates="paciente", cascade="all, delete-orphan"
    )
    exames = relationship(
        "ExamePreventivo", back_populates="paciente", cascade="all, delete-orphan"
    )
    ciclos_menstruais = relationship(
        "CicloMenstrual", back_populates="paciente", cascade="all, delete-orphan"
    )
    triagens_vd = relationship(
        "TriagemViolencia", back_populates="paciente", cascade="all, delete-orphan"
    )


class ProntuarioGinecologico(Base):
    __tablename__ = "prontuarios_ginecologicos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False, index=True)
    data_consulta = Column(Date, nullable=False)
    queixas = Column(Text)
    # Histórico obstétrico: G (gestações) P (partos) A (abortos)
    gestacoes = Column(Integer, default=0)
    partos_normais = Column(Integer, default=0)
    partos_cesareos = Column(Integer, default=0)
    abortos = Column(Integer, default=0)
    metodo_contraceptivo = Column(String(150))
    ultima_menstruacao = Column(Date)
    historico_dst = Column(Text)
    alergias = Column(Text)
    medicamentos_uso = Column(Text)
    observacoes = Column(Text)
    medico_responsavel = Column(String(200))

    paciente = relationship("Paciente", back_populates="prontuarios")


class ExamePreventivo(Base):
    __tablename__ = "exames_preventivos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False, index=True)
    tipo_exame = Column(String(60), nullable=False)
    data_realizacao = Column(Date)
    resultado = Column(Text)
    resultado_alterado = Column(Boolean, default=False)
    proximo_previsto = Column(Date)
    laboratorio = Column(String(200))
    medico_solicitante = Column(String(200))
    observacoes = Column(Text)

    paciente = relationship("Paciente", back_populates="exames")


class CicloMenstrual(Base):
    __tablename__ = "ciclos_menstruais"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False, index=True)
    data_inicio = Column(Date, nullable=False)
    data_fim = Column(Date)
    duracao_dias = Column(Integer)
    intensidade = Column(String(20))       # leve | moderada | intensa | muito_intensa
    dor_escala = Column(Integer)           # 0-10
    sintomas_associados = Column(Text)     # cólica, cefaleia, humor, etc.
    observacoes = Column(Text)

    paciente = relationship("Paciente", back_populates="ciclos_menstruais")


class TriagemViolencia(Base):
    """Registro confidencial de triagem de violência doméstica.
    Acesso restrito por controle de permissão na camada de aplicação.
    """
    __tablename__ = "triagens_violencia"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paciente_id = Column(Integer, ForeignKey("pacientes.id"), nullable=False, index=True)
    data_triagem = Column(Date, nullable=False)
    nivel_risco = Column(String(20), nullable=False)
    indicadores_identificados = Column(Text)    # JSON serializado
    protocolo_acionado = Column(Boolean, default=False)
    encaminhamentos_realizados = Column(Text)
    plano_seguranca = Column(Text)
    observacoes_confidenciais = Column(Text)

    paciente = relationship("Paciente", back_populates="triagens_vd")


class Medicamento(Base):
    __tablename__ = "medicamentos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nome_comercial = Column(String(200), nullable=False)
    principio_ativo = Column(String(200), nullable=False)
    categoria = Column(String(100))        # anticoncepcional | hormonal | antibiotico | etc.
    indicacoes = Column(Text)
    contraindicacoes = Column(Text)
    seguro_gestacao = Column(Boolean)
    seguro_amamentacao = Column(Boolean)
    interacoes_importantes = Column(Text)
    observacoes = Column(Text)


class ProtocoloMedico(Base):
    __tablename__ = "protocolos_medicos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    titulo = Column(String(300), nullable=False)
    categoria = Column(String(100), index=True)
    palavras_chave = Column(Text)          # separadas por vírgula
    conteudo = Column(Text, nullable=False)
    fonte = Column(String(300))
    atualizado_em = Column(Date)


# ─────────────────────────────────────────────
# Schemas Pydantic
# ─────────────────────────────────────────────

class PacienteSchema(BaseModel):
    id: Optional[int] = None
    nome: str
    data_nascimento: date
    cpf: Optional[str] = None
    telefone: Optional[str] = None

    model_config = {"from_attributes": True}


class ProntuarioSchema(BaseModel):
    paciente_id: int
    data_consulta: date
    queixas: Optional[str] = None
    gestacoes: int = 0
    partos_normais: int = 0
    partos_cesareos: int = 0
    abortos: int = 0
    metodo_contraceptivo: Optional[str] = None
    observacoes: Optional[str] = None

    model_config = {"from_attributes": True}


class ExamePreventivoSchema(BaseModel):
    paciente_id: int
    tipo_exame: str
    data_realizacao: Optional[date] = None
    resultado: Optional[str] = None
    resultado_alterado: bool = False
    proximo_previsto: Optional[date] = None

    model_config = {"from_attributes": True}


class TriagemSintomasInput(BaseModel):
    sintomas: list[str] = Field(description="Lista de sintomas relatados pela paciente")
    duracao_sintomas: Optional[str] = Field(None, description="Duração dos sintomas (ex: '3 dias', '2 semanas')")
    intensidade: Optional[int] = Field(None, ge=0, le=10, description="Intensidade 0-10")
    historico_relevante: Optional[str] = Field(None, description="Informações relevantes do histórico")


class TriagemVDInput(BaseModel):
    respostas_wast: list[bool] = Field(
        description="Respostas ao WAST (Woman Abuse Screening Tool) - 8 perguntas True/False"
    )
    sinais_fisicos: Optional[list[str]] = Field(None, description="Sinais físicos observados")
    observacoes: Optional[str] = Field(None, description="Observações clínicas adicionais")
