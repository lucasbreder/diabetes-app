"""
Assistente Médica Especializada em Saúde da Mulher
Pipeline LangChain com integração a bases de dados clínicas especializadas.
"""

from .pipeline import criar_pipeline_assistente
from .database import init_db

__all__ = ["criar_pipeline_assistente", "init_db"]
