"""
Protocolos de segurança específicos (item 4 – seção 2).
Verificação de identidade, criptografia de dados VD, alertas e emergências.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Chave Fernet: env ou arquivo local (não versionar)
_KEY_PATH = Path("data/.vd_encryption_key")


def _obter_fernet():
    from cryptography.fernet import Fernet

    chave = os.environ.get("VD_ENCRYPTION_KEY", "").strip()
    if not chave:
        _KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _KEY_PATH.exists():
            chave = _KEY_PATH.read_text().strip()
        else:
            chave = Fernet.generate_key().decode()
            _KEY_PATH.write_text(chave)
            _KEY_PATH.chmod(0o600)
    return Fernet(chave.encode() if isinstance(chave, str) else chave)


def criptografar_texto_vd(texto: str) -> str:
    """Criptografia em repouso para observações e indicadores de violência doméstica."""
    if not texto or not texto.strip():
        return ""
    f = _obter_fernet()
    return "ENC:" + f.encrypt(texto.encode("utf-8")).decode("ascii")


def descriptografar_texto_vd(texto_cifrado: str) -> str:
    if not texto_cifrado or not texto_cifrado.startswith("ENC:"):
        return texto_cifrado or ""
    f = _obter_fernet()
    return f.decrypt(texto_cifrado[4:].encode("ascii")).decode("utf-8")


# PIN de demonstração; em produção usar autenticação institucional (LDAP/OAuth)
PIN_PROFISSIONAL_PADRAO = os.environ.get("ASSISTENTE_PIN_PROFISSIONAL", "saudemulher2026")


def verificar_identidade_profissional(pin_informado: str) -> bool:
    """Verificação de identidade para áreas sensíveis (VD, dados críticos)."""
    if not pin_informado:
        return False
    esperado = PIN_PROFISSIONAL_PADRAO.strip()
    return hashlib.sha256(pin_informado.encode()).hexdigest() == hashlib.sha256(
        esperado.encode()
    ).hexdigest()


_PADROES_CONTEXTO_SENSIVEL = re.compile(
    r"(viol[êe]ncia|abuso|maus-tratos|suic[íi]d\w*|automutila[çc][ãa]o|"
    r"sa[úu]de mental|amea[çc]a (do parceiro|f[íi]sica)|agress[ãa]o)",
    re.IGNORECASE,
)


def contexto_requer_verificacao(
    fluxo: str,
    mensagem: str = "",
    nivel_risco_vd: Optional[str] = None,
) -> bool:
    """Indica se o acesso exige verificação de identidade prévia."""
    if fluxo in ("vd", "triagem_vd", "acesso_prontuario_vd"):
        return True
    if nivel_risco_vd in ("alto", "critico"):
        return True
    return bool(mensagem and _PADROES_CONTEXTO_SENSIVEL.search(mensagem))


@dataclass
class ResultadoEmergencia:
    nivel: str  # nenhum | moderado | alto | critico | emergencia
    protocolo: str
    mensagem_equipe: str
    acionar_alerta: bool


_PADROES_EMERGENCIA = re.compile(
    r"\b(parada cardíaca|parada cardiaca|convulsão|convulsao|inconsciente|"
    r"sangramento (intenso|abundante|incontrolável)|eclampsia|trabalho de parto prematuro|"
    r"dor (torácica|toracica) súbita|samu|emergência imediata)\b",
    re.IGNORECASE,
)
_PADROES_CRITICO = re.compile(
    r"\b(risco imediato|ameaça de morte|estrangulamento|arma|faca|"
    r"violência grave|agressor presente)\b",
    re.IGNORECASE,
)


def avaliar_situacao_emergencia(
    mensagem: str = "",
    nivel_risco_vd: Optional[str] = None,
    sintomas: Optional[list[str]] = None,
) -> ResultadoEmergencia:
    """Protocolos de emergência para situações críticas."""
    texto = " ".join(filter(None, [mensagem, " ".join(sintomas or [])]))

    if _PADROES_EMERGENCIA.search(texto):
        return ResultadoEmergencia(
            nivel="emergencia",
            protocolo="EMERG_OBSTETRICA_GERAL",
            mensagem_equipe="Situação de emergência detectada — acionar equipe e SAMU 192.",
            acionar_alerta=True,
        )

    if nivel_risco_vd == "critico" or _PADROES_CRITICO.search(texto):
        return ResultadoEmergencia(
            nivel="critico",
            protocolo="VD_RISCO_CRITICO",
            mensagem_equipe="Risco crítico de violência — acionar equipe multidisciplinar e rede de proteção.",
            acionar_alerta=True,
        )

    if nivel_risco_vd == "alto":
        return ResultadoEmergencia(
            nivel="alto",
            protocolo="VD_RISCO_ALTO",
            mensagem_equipe="Risco alto de violência — priorizar acolhimento e plano de segurança.",
            acionar_alerta=True,
        )

    if nivel_risco_vd == "moderado":
        return ResultadoEmergencia(
            nivel="moderado",
            protocolo="VD_ACOLHIMENTO",
            mensagem_equipe="Risco moderado — reforçar triagem e encaminhamento.",
            acionar_alerta=False,
        )

    return ResultadoEmergencia(
        nivel="nenhum",
        protocolo="",
        mensagem_equipe="",
        acionar_alerta=False,
    )


def validar_estabilidade_resposta(
    texto: str,
    mensagem_usuario: str = "",
    nivel_risco_vd: Optional[str] = None,
    contexto_vd: bool = False,
) -> tuple[str, list[str]]:
    """
    Validação da resposta do LLM antes do retorno (estabilidade e previsibilidade).
    Retorna texto validado e lista de ajustes aplicados.
    """
    from .safety import aplicar_guardrails_resposta

    ajustes: list[str] = []
    if not texto or len(texto.strip()) < 10:
        ajustes.append("Resposta vazia ou muito curta — substituída por mensagem padrão.")
        texto = (
            "Não foi possível gerar uma orientação clínica estável neste momento. "
            "Consulte o protocolo institucional ou um especialista presencialmente."
        )

    if re.search(r"\b(receita|prescrição médica)\s*:\s*", texto, re.IGNORECASE):
        ajustes.append("Trecho com formato de receita removido da resposta automática.")
        texto = re.sub(
            r"(receita|prescrição médica)\s*:\s*[^\n]+",
            "[Indicação medicamentosa — validar com especialista]",
            texto,
            flags=re.IGNORECASE,
        )

    texto_final = aplicar_guardrails_resposta(
        texto,
        mensagem_usuario,
        nivel_risco_vd=nivel_risco_vd,
        contexto_vd=contexto_vd,
    )
    if texto_final != texto:
        ajustes.append("Guardrails de segurança aplicados na pós-validação.")

    return texto_final, ajustes
