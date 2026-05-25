"""
Utilitário compartilhado para chamadas ao LLM (Ollama) nos fluxos LangGraph.

Centraliza a comunicação com o modelo para facilitar troca de modelo,
configuração de temperatura e tratamento de erros.
"""

from __future__ import annotations

import os
from typing import Optional

import ollama

from medical_assistant.safety import REGRAS_SEGURANCA_PROMPT
from medical_assistant.validation_pipeline import processar_resposta_final

# Preferência de modelos (primeiro encontrado no Ollama local é usado)
MODELO_PADRAO = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
MODELOS_PREFERIDOS = (
    "llama3.2:1b",
    "llama3:latest",
    "llama3",
    "gemma2:2b",
    "mistral:7b",
)


def _listar_modelos_locais() -> list[str]:
    try:
        return [m.model for m in ollama.list().models]
    except Exception:
        return []


def resolver_modelo(modelo: Optional[str] = None) -> str:
    """
    Retorna um modelo instalado no Ollama.
    Se o solicitado não existir, usa o primeiro da lista de preferência disponível.
    """
    instalados = _listar_modelos_locais()
    if not instalados:
        return modelo or MODELO_PADRAO

    if modelo and any(modelo in m or m.startswith(modelo) for m in instalados):
        return next(m for m in instalados if modelo in m or m.startswith(modelo))

    for preferido in MODELOS_PREFERIDOS:
        for instalado in instalados:
            if preferido in instalado or instalado.startswith(preferido.split(":")[0]):
                return instalado

    return instalados[0]


def consultar_llm(
    prompt: str,
    modelo: str = MODELO_PADRAO,
    system_prompt: Optional[str] = None,
    contexto_guardrail: str = "",
    *,
    fluxo: str = "langgraph",
    especialidade: str = "ginecologia",
    contexto_paciente: str = "",
    protocolos_contexto: str = "",
    nivel_risco_vd: Optional[str] = None,
    paciente_id: Optional[int] = None,
    incluir_explicabilidade: bool = True,
) -> str:
    """
    Envia um prompt ao LLM via Ollama e retorna a resposta como string.

    Args:
        prompt: O prompt do usuário.
        modelo: Nome do modelo Ollama a utilizar.
        system_prompt: Prompt de sistema opcional para definir o papel da IA.
        contexto_guardrail: Texto da pergunta/caso para pós-validação (item 4).
        fluxo: Identificador do fluxo ("vd", "obstetrico", "triagem", "prevencao", ...).
        especialidade: Especialidade clínica do fluxo.
        contexto_paciente: Resumo clínico da paciente (para explicabilidade).
        protocolos_contexto: Texto com protocolos relevantes (será auto-buscado se vazio).
        nivel_risco_vd: Nível de risco quando o fluxo é VD.
        paciente_id: ID da paciente (para auditoria).
        incluir_explicabilidade: Se True, adiciona bloco de explicabilidade ao retorno.

    Returns:
        A resposta do modelo como string, ou mensagem de fallback em caso de erro.
    """
    modelo_efetivo = resolver_modelo(modelo)
    messages = []

    sistema = (system_prompt or "").strip()
    if REGRAS_SEGURANCA_PROMPT.strip() not in sistema:
        sistema = f"{sistema}\n\n{REGRAS_SEGURANCA_PROMPT}".strip()

    if sistema:
        messages.append({"role": "system", "content": sistema})

    messages.append({"role": "user", "content": prompt})

    try:
        response = ollama.chat(model=modelo_efetivo, messages=messages)
        texto = response.message.content
        resposta, _ = processar_resposta_final(
            texto,
            mensagem_usuario=contexto_guardrail or prompt,
            paciente_id=paciente_id,
            fluxo=fluxo,
            especialidade=especialidade,
            contexto_paciente=contexto_paciente,
            protocolos_contexto=protocolos_contexto,
            nivel_risco_vd=nivel_risco_vd,
            incluir_explicabilidade=incluir_explicabilidade,
        )
        return resposta
    except Exception as e:
        instalados = _listar_modelos_locais()
        dica = (
            f" Modelos locais: {', '.join(instalados)}."
            if instalados
            else " Rode: ollama pull llama3  (ou ollama pull llama3.2:1b)"
        )
        fallback = (
            f"[LLM indisponível — resposta baseada em regras] "
            f"Erro: {e}.{dica}"
        )
        print(f"⚠️  Falha na comunicação com o LLM ({modelo_efetivo}): {e}")
        return fallback
