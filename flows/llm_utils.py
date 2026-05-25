"""
Utilitário compartilhado para chamadas ao LLM (Ollama) nos fluxos LangGraph.

Centraliza a comunicação com o modelo para facilitar troca de modelo,
configuração de temperatura e tratamento de erros.
"""

from typing import Optional

import ollama

# Modelo padrão para todos os fluxos
MODELO_PADRAO = "llama3.2:1b"


def consultar_llm(
    prompt: str,
    modelo: str = MODELO_PADRAO,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Envia um prompt ao LLM via Ollama e retorna a resposta como string.

    Args:
        prompt: O prompt do usuário.
        modelo: Nome do modelo Ollama a utilizar.
        system_prompt: Prompt de sistema opcional para definir o papel da IA.

    Returns:
        A resposta do modelo como string, ou mensagem de fallback em caso de erro.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    try:
        response = ollama.chat(model=modelo, messages=messages)
        return response.message.content
    except Exception as e:
        fallback = (
            f"[LLM indisponível — resposta baseada em regras] "
            f"Erro: {e}"
        )
        print(f"⚠️  Falha na comunicação com o LLM: {e}")
        return fallback
