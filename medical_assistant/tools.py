"""
LangChain Tools que encapsulam as consultas ao banco de dados clínico.
Essas ferramentas são usadas pelo agente ReAct para responder perguntas contextualizadas.
"""

from __future__ import annotations

from datetime import date
from typing import List

from langchain_core.tools import tool

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
)


def _formatar_exame(e) -> str:
    atrasado = ""
    if e.proximo_previsto and e.proximo_previsto < date.today():
        dias = (date.today() - e.proximo_previsto).days
        atrasado = f" ⚠️ ATRASADO HÁ {dias} DIAS"
    return (
        f"- Tipo: {e.tipo_exame}\n"
        f"  Realizado em: {e.data_realizacao or 'N/A'}\n"
        f"  Resultado: {e.resultado or 'N/A'}{'  ⚠️ ALTERADO' if e.resultado_alterado else ''}\n"
        f"  Próximo previsto: {e.proximo_previsto or 'N/A'}{atrasado}\n"
        f"  Lab: {e.laboratorio or 'N/A'}"
    )


def criar_ferramentas(paciente_id: int) -> List:
    """
    Factory que cria as ferramentas LangChain com o paciente_id configurado via closure.
    """

    @tool
    def obter_dados_paciente(_: str = "") -> str:
        """Retorna os dados cadastrais e demográficos da paciente atual."""
        p = buscar_paciente(paciente_id)
        if not p:
            return "Paciente não encontrada no banco de dados."
        hoje = date.today()
        idade = hoje.year - p.data_nascimento.year - (
            (hoje.month, hoje.day) < (p.data_nascimento.month, p.data_nascimento.day)
        )
        return (
            f"Nome: {p.nome}\n"
            f"Data de nascimento: {p.data_nascimento} ({idade} anos)\n"
            f"Telefone: {p.telefone or 'N/A'}"
        )

    @tool
    def obter_prontuario_ginecologico(_: str = "") -> str:
        """Retorna o histórico de prontuários ginecológicos/obstétricos da paciente,
        incluindo queixas, histórico obstétrico (G_P_A), método contraceptivo e observações."""
        prontuarios = buscar_prontuarios(paciente_id)
        if not prontuarios:
            return "Nenhum prontuário ginecológico registrado para esta paciente."
        partes = []
        for p in prontuarios[:3]:  # últimos 3 prontuários
            partes.append(
                f"Data consulta: {p.data_consulta}\n"
                f"Queixas: {p.queixas or 'N/A'}\n"
                f"Histórico obstétrico: G{p.gestacoes}P{p.partos_normais+p.partos_cesareos}A{p.abortos} "
                f"({p.partos_normais} parto(s) normal(is), {p.partos_cesareos} cesárea(s))\n"
                f"Método contraceptivo: {p.metodo_contraceptivo or 'N/A'}\n"
                f"Última menstruação: {p.ultima_menstruacao or 'N/A'}\n"
                f"Alergias: {p.alergias or 'Nenhuma'}\n"
                f"Medicamentos em uso: {p.medicamentos_uso or 'Nenhum'}\n"
                f"Histórico IST: {p.historico_dst or 'Nenhum relatado'}\n"
                f"Observações: {p.observacoes or 'N/A'}\n"
                f"Responsável: {p.medico_responsavel or 'N/A'}"
            )
        return "\n\n---\n\n".join(partes)

    @tool
    def obter_historico_exames(_: str = "") -> str:
        """Retorna o histórico completo de exames preventivos da paciente
        (papanicolau, mamografia, ultrassom pélvico, densitometria, etc.)."""
        exames = buscar_exames(paciente_id)
        if not exames:
            return "Nenhum exame preventivo registrado."
        return "\n\n".join(_formatar_exame(e) for e in exames)

    @tool
    def verificar_exames_atrasados(_: str = "") -> str:
        """Verifica se há exames preventivos com prazo vencido para a paciente.
        Retorna lista de exames atrasados com número de dias em atraso."""
        atrasados = buscar_exames_atrasados(paciente_id)
        alterados = buscar_exames_alterados(paciente_id)

        partes = []
        if atrasados:
            partes.append("🔴 EXAMES COM PRAZO VENCIDO:")
            for e in atrasados:
                dias = (date.today() - e.proximo_previsto).days
                partes.append(f"  • {e.tipo_exame.upper()}: vencido há {dias} dias (previsto: {e.proximo_previsto})")

        if alterados:
            partes.append("\n⚠️ EXAMES COM RESULTADO ALTERADO:")
            for e in alterados:
                partes.append(f"  • {e.tipo_exame.upper()} ({e.data_realizacao}): {e.resultado[:100]}...")

        return "\n".join(partes) if partes else "✅ Todos os exames preventivos estão em dia."

    @tool
    def obter_calendario_menstrual(_: str = "") -> str:
        """Retorna o histórico menstrual recente da paciente (últimos 6 ciclos),
        incluindo datas, duração, intensidade, dor e sintomas associados."""
        ciclos = buscar_ciclos_menstruais(paciente_id, ultimos_n=6)
        if not ciclos:
            return "Nenhum registro menstrual disponível."

        partes = []
        duracoes = [c.duracao_dias for c in ciclos if c.duracao_dias]
        media_duracao = sum(duracoes) / len(duracoes) if duracoes else None

        for c in ciclos:
            partes.append(
                f"• Início: {c.data_inicio} | Duração: {c.duracao_dias or 'N/A'} dias | "
                f"Intensidade: {c.intensidade or 'N/A'} | Dor: {c.dor_escala or 'N/A'}/10\n"
                f"  Sintomas: {c.sintomas_associados or 'Nenhum'}"
            )

        resultado = "\n".join(partes)
        if media_duracao:
            resultado += f"\n\nMédia de duração dos ciclos: {media_duracao:.1f} dias"

        # Calcular intervalo entre ciclos
        if len(ciclos) >= 2:
            intervalos = []
            datas = sorted([c.data_inicio for c in ciclos])
            for i in range(1, len(datas)):
                intervalos.append((datas[i] - datas[i - 1]).days)
            if intervalos:
                media_intervalo = sum(intervalos) / len(intervalos)
                resultado += f"\nMédia de intervalo entre ciclos: {media_intervalo:.0f} dias"

        return resultado

    @tool
    def obter_triagem_violencia(_: str = "") -> str:
        """[CONFIDENCIAL] Retorna o histórico de triagens de violência doméstica da paciente.
        Use apenas quando relevante clinicamente. Requer tratamento com máxima confidencialidade."""
        triagens = buscar_triagens_vd(paciente_id)
        if not triagens:
            return "Nenhuma triagem de violência doméstica registrada."
        partes = []
        for t in triagens:
            partes.append(
                f"Data: {t.data_triagem}\n"
                f"Nível de risco: {t.nivel_risco.upper()}\n"
                f"Indicadores: {t.indicadores_identificados or 'N/A'}\n"
                f"Protocolo acionado: {'Sim' if t.protocolo_acionado else 'Não'}\n"
                f"Encaminhamentos: {t.encaminhamentos_realizados or 'N/A'}"
            )
        return "\n---\n".join(partes)

    @tool
    def buscar_informacoes_medicamento(nome_medicamento: str) -> str:
        """Busca informações sobre um medicamento por nome comercial ou princípio ativo.
        Retorna indicações, contraindicações e segurança na gestação/amamentação.
        
        Args:
            nome_medicamento: Nome comercial ou princípio ativo do medicamento a pesquisar.
        """
        medicamentos = buscar_medicamento(nome_medicamento)
        if not medicamentos:
            return f"Medicamento '{nome_medicamento}' não encontrado na base de dados."
        partes = []
        for m in medicamentos[:3]:
            partes.append(
                f"Nome: {m.nome_comercial}\n"
                f"Princípio ativo: {m.principio_ativo}\n"
                f"Categoria: {m.categoria}\n"
                f"Indicações: {m.indicacoes}\n"
                f"Contraindicações: {m.contraindicacoes}\n"
                f"Seguro na gestação: {'Sim' if m.seguro_gestacao else 'Não' if m.seguro_gestacao is not None else 'Avaliar com médico'}\n"
                f"Seguro na amamentação: {'Sim' if m.seguro_amamentacao else 'Não' if m.seguro_amamentacao is not None else 'Avaliar com médico'}\n"
                f"Interações importantes: {m.interacoes_importantes or 'N/A'}\n"
                f"Observações: {m.observacoes or 'N/A'}"
            )
        return "\n\n---\n\n".join(partes)

    @tool
    def consultar_protocolo_medico(consulta: str) -> str:
        """Consulta a base de protocolos médicos especializados por tema ou categoria.
        Retorna protocolos de sociedades médicas (FEBRASGO, INCA, Ministério da Saúde).
        
        Args:
            consulta: Tema ou palavras-chave para buscar no protocolo (ex: 'papanicolau', 'endometriose', 'violência doméstica').
        """
        protocolos = buscar_protocolos(termo=consulta)
        if not protocolos:
            return f"Nenhum protocolo encontrado para '{consulta}'."
        partes = []
        for p in protocolos[:2]:  # máximo 2 protocolos para não sobrecarregar o contexto
            partes.append(
                f"📋 {p.titulo}\n"
                f"Fonte: {p.fonte}\n"
                f"Atualizado em: {p.atualizado_em}\n\n"
                f"{p.conteudo}"
            )
        return "\n\n{'='*60}\n\n".join(partes)

    return [
        obter_dados_paciente,
        obter_prontuario_ginecologico,
        obter_historico_exames,
        verificar_exames_atrasados,
        obter_calendario_menstrual,
        obter_triagem_violencia,
        buscar_informacoes_medicamento,
        consultar_protocolo_medico,
    ]
