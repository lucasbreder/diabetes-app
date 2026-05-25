"""
Logging e auditoria especializados (item 4 – seção 3).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any, Optional

from .database import get_session
from .models import AlertaSeguranca, LogAcessoSensivel, LogInteracao


def _resumir(texto: str, max_len: int = 480) -> str:
    t = (texto or "").strip().replace("\n", " ")
    return t[:max_len] + ("…" if len(t) > max_len else "")


def registrar_interacao(
    *,
    fluxo: str,
    especialidade: str,
    mensagem: str = "",
    resposta: str = "",
    paciente_id: Optional[int] = None,
    guardrails_aplicados: bool = False,
    caso_violencia: bool = False,
    metadados: Optional[dict[str, Any]] = None,
) -> int:
    """Rastreamento detalhado de interações com o assistente."""
    with get_session() as s:
        log = LogInteracao(
            paciente_id=paciente_id,
            especialidade=especialidade,
            fluxo=fluxo,
            mensagem_resumo=_resumir(mensagem),
            resposta_resumo=_resumir(resposta),
            guardrails_aplicados=guardrails_aplicados,
            caso_violencia=caso_violencia,
            metadados_json=json.dumps(metadados or {}, ensure_ascii=False),
        )
        s.add(log)
        s.flush()
        return log.id


def registrar_log_violencia(
    *,
    paciente_id: Optional[int],
    acao: str,
    profissional_id: str = "sessao_streamlit",
    detalhes: Optional[dict[str, Any]] = None,
) -> int:
    """Log específico para casos de violência doméstica."""
    with get_session() as s:
        log = LogAcessoSensivel(
            tipo_dado="violencia_domestica",
            paciente_id=paciente_id,
            acao=acao,
            profissional_id=profissional_id,
        )
        s.add(log)
        s.flush()
        log_id = log.id

    meta = {"log_acesso_id": log_id, **(detalhes or {})}
    registrar_interacao(
        fluxo="vd",
        especialidade="violencia_domestica",
        mensagem=detalhes.get("resumo", "") if detalhes else "",
        paciente_id=paciente_id,
        caso_violencia=True,
        metadados=meta,
    )
    return log_id


def registrar_acesso_sensivel(
    tipo_dado: str,
    acao: str,
    paciente_id: Optional[int] = None,
    profissional_id: str = "sessao_streamlit",
) -> int:
    """Auditoria de acesso a dados sensíveis."""
    with get_session() as s:
        log = LogAcessoSensivel(
            tipo_dado=tipo_dado,
            paciente_id=paciente_id,
            acao=acao,
            profissional_id=profissional_id,
        )
        s.add(log)
        s.flush()
        return log.id


def registrar_alerta_seguranca(
    *,
    paciente_id: Optional[int],
    nivel: str,
    motivo: str,
    protocolo_emergencia: str = "",
) -> int:
    with get_session() as s:
        alerta = AlertaSeguranca(
            paciente_id=paciente_id,
            nivel=nivel,
            motivo=motivo,
            protocolo_emergencia=protocolo_emergencia or None,
        )
        s.add(alerta)
        s.flush()
        return alerta.id


def listar_alertas_pendentes(limite: int = 20) -> list[AlertaSeguranca]:
    with get_session() as s:
        return (
            s.query(AlertaSeguranca)
            .filter(AlertaSeguranca.resolvido.is_(False))
            .order_by(AlertaSeguranca.criado_em.desc())
            .limit(limite)
            .all()
        )


def listar_logs_acesso_sensivel(limite: int = 30) -> list[LogAcessoSensivel]:
    with get_session() as s:
        return (
            s.query(LogAcessoSensivel)
            .order_by(LogAcessoSensivel.criado_em.desc())
            .limit(limite)
            .all()
        )


def listar_logs_interacao(
    especialidade: Optional[str] = None,
    fluxo: Optional[str] = None,
    apenas_vd: bool = False,
    limite: int = 50,
) -> list[LogInteracao]:
    with get_session() as s:
        q = s.query(LogInteracao).order_by(LogInteracao.criado_em.desc())
        if especialidade:
            q = q.filter(LogInteracao.especialidade == especialidade)
        if fluxo:
            q = q.filter(LogInteracao.fluxo == fluxo)
        if apenas_vd:
            q = q.filter(LogInteracao.caso_violencia.is_(True))
        return q.limit(limite).all()


def relatorio_utilizacao_por_especialidade() -> dict[str, int]:
    """Relatórios de utilização por especialidade médica."""
    with get_session() as s:
        logs = s.query(LogInteracao.especialidade).all()
    contagem = Counter(row[0] for row in logs)
    return dict(contagem.most_common())


def relatorio_resumo_auditoria() -> dict[str, Any]:
    with get_session() as s:
        total_interacoes = s.query(LogInteracao).count()
        total_vd = s.query(LogInteracao).filter(LogInteracao.caso_violencia.is_(True)).count()
        total_acessos = s.query(LogAcessoSensivel).count()
        alertas_pendentes = (
            s.query(AlertaSeguranca).filter(AlertaSeguranca.resolvido.is_(False)).count()
        )
    return {
        "gerado_em": datetime.utcnow().isoformat(),
        "total_interacoes": total_interacoes,
        "interacoes_violencia_domestica": total_vd,
        "acessos_dados_sensiveis": total_acessos,
        "alertas_seguranca_pendentes": alertas_pendentes,
        "por_especialidade": relatorio_utilizacao_por_especialidade(),
    }
