"""
Script de demonstração dos 4 fluxos LangGraph.

Executa cada fluxo com dados de exemplo e exibe os resultados.
Pode ser rodado com: python demo_flows.py
"""

import json
import sys
from datetime import datetime


def demo_triagem_ginecologica():
    """Demonstra o fluxo de triagem ginecológica."""
    from flows.triagem_ginecologica import criar_fluxo_triagem_ginecologica

    print("\n" + "=" * 60)
    print("  🩺 FLUXO 1 — TRIAGEM GINECOLÓGICA")
    print("=" * 60)

    fluxo = criar_fluxo_triagem_ginecologica()

    entrada = {
        "paciente_id": "PAC-001",
        "nome_paciente": "Maria Silva",
        "idade": 35,
        "sintomas": ["dor pélvica", "corrimento anormal", "irregularidade menstrual"],
        "historico_menstrual": "Ciclos irregulares nos últimos 3 meses",
        "uso_contraceptivo": "DIU de cobre",
        "gestacoes_anteriores": 2,
        "ultima_consulta_gineco": "2024-06-15",
        "historico_familiar": ["câncer de mama (mãe)", "mioma uterino (irmã)"],
        "queixas_adicionais": "Cansaço frequente",
    }

    print(f"\n📋 Paciente: {entrada['nome_paciente']}, {entrada['idade']} anos")
    print(f"   Sintomas: {', '.join(entrada['sintomas'])}")
    print("\n⏳ Executando fluxo...")

    resultado = fluxo.invoke(entrada)

    print(f"\n✅ Classificação: {resultado.get('classificacao_urgencia', {}).get('codigo', 'N/A')}")
    print(f"   Descrição: {resultado.get('classificacao_urgencia', {}).get('descricao', '')}")
    print(f"   Exames sugeridos: {len(resultado.get('exames_sugeridos', []))}")
    for e in resultado.get("exames_sugeridos", []):
        print(f"     • {e['nome']} ({e['prioridade']})")
    print(f"   Agendamento: {resultado.get('agendamento', {}).get('data_sugerida', '')}")
    print(f"\n{resultado.get('resumo_final', '')}")
    return resultado


def demo_violencia_domestica():
    """Demonstra o fluxo de detecção de violência doméstica."""
    from flows.violencia_domestica import criar_fluxo_violencia_domestica

    print("\n" + "=" * 60)
    print("  🛡️ FLUXO 2 — DETECÇÃO DE VIOLÊNCIA DOMÉSTICA")
    print("=" * 60)

    fluxo = criar_fluxo_violencia_domestica()

    entrada = {
        "paciente_id": "PAC-002",
        "nome_paciente": "Ana Oliveira",
        "idade": 28,
        "sinais_alerta": [
            "lesões em diferentes estágios de cicatrização",
            "relato inconsistente sobre lesões",
        ],
        "relato_paciente": "Diz que caiu da escada, mas lesões são incompatíveis",
        "lesoes_observadas": ["equimose em braço esquerdo", "escoriação na face"],
        "historico_atendimentos": 3,
        "acompanhante_presente": True,
        "comportamento_observado": [
            "paciente evita contato visual",
            "comportamento ansioso ou submisso",
        ],
    }

    print(f"\n📋 Paciente: {entrada['nome_paciente']}, {entrada['idade']} anos")
    print(f"   Sinais: {', '.join(entrada['sinais_alerta'][:2])}")
    print("\n⏳ Executando fluxo...")

    resultado = fluxo.invoke(entrada)

    print(f"\n🚨 Nível de risco: {resultado.get('nivel_risco', 'N/A').upper()}")
    print(f"   Score: {resultado.get('avaliacao_risco', {}).get('score', 0)}")
    protocolo = resultado.get("protocolo_seguranca", {})
    print(f"   Protocolo: {protocolo.get('prioridade', '')}")
    print(f"   Ação imediata: {protocolo.get('acao_imediata', '')}")
    equipe = resultado.get("equipe_acionada", {})
    print(f"   Equipe: {', '.join(equipe.get('profissionais', []))}")
    print(f"   Notificação compulsória: {'Sim' if equipe.get('notificacao_compulsoria') else 'Não'}")
    seguimento = resultado.get("plano_seguimento", {})
    print(f"   Retorno em: {seguimento.get('retorno_em', '')}")
    print(f"\n{resultado.get('resumo_final', '')}")
    return resultado


def demo_obstetrico():
    """Demonstra o fluxo obstétrico."""
    from flows.obstetrico import criar_fluxo_obstetrico

    print("\n" + "=" * 60)
    print("  🤰 FLUXO 3 — OBSTÉTRICO")
    print("=" * 60)

    fluxo = criar_fluxo_obstetrico()

    entrada = {
        "paciente_id": "PAC-003",
        "nome_paciente": "Juliana Santos",
        "idade": 32,
        "semanas_gestacao": 24,
        "tipo_gestacao": "única",
        "gestacoes_anteriores": 1,
        "partos_anteriores": 1,
        "abortos_anteriores": 0,
        "comorbidades": ["hipotireoidismo"],
        "medicamentos_em_uso": ["levotiroxina 50mcg"],
        "pressao_arterial": "120/80",
        "peso_atual": 72.5,
        "altura": 1.65,
        "glicemia_jejum": 95.0,
        "grupo_sanguineo": "O+",
        "queixas_atuais": ["edema em membros inferiores", "dor lombar"],
    }

    print(f"\n📋 Gestante: {entrada['nome_paciente']}, {entrada['idade']} anos")
    print(f"   IG: {entrada['semanas_gestacao']} semanas")
    print(f"   Queixas: {', '.join(entrada['queixas_atuais'])}")
    print("\n⏳ Executando fluxo...")

    resultado = fluxo.invoke(entrada)

    print(f"\n✅ Risco gestacional: {resultado.get('nivel_risco', 'N/A').upper()}")
    print(f"   Exames agendados: {len(resultado.get('exames_agendados', []))}")
    for e in resultado.get("exames_agendados", [])[:5]:
        print(f"     • {e['nome']} ({e['prazo']})")
    alertas = resultado.get("alertas_urgencia", [])
    if alertas:
        print(f"   ⚠️ Alertas: {len(alertas)}")
        for a in alertas:
            print(f"     {a}")
    acomp = resultado.get("plano_acompanhamento", {})
    print(f"   Próxima consulta: {acomp.get('proxima_consulta', '')}")
    print(f"   Frequência: {acomp.get('frequencia_consultas', '')}")
    print(f"\n{resultado.get('resumo_final', '')}")
    return resultado


def demo_prevencao():
    """Demonstra o fluxo de prevenção."""
    from flows.prevencao import criar_fluxo_prevencao

    print("\n" + "=" * 60)
    print("  💊 FLUXO 4 — PREVENÇÃO")
    print("=" * 60)

    fluxo = criar_fluxo_prevencao()

    entrada = {
        "paciente_id": "PAC-004",
        "nome_paciente": "Fernanda Costa",
        "idade": 52,
        "sexo": "F",
        "historico_pessoal": ["hipertensão controlada"],
        "historico_familiar": ["câncer de mama (mãe)", "diabetes tipo 2 (pai)"],
        "ultimo_papanicolau": "2023-03-10",
        "ultima_mamografia": "2022-08-20",
        "ultima_densitometria": "nunca",
        "ultimo_check_up": "2024-01-15",
        "vacinas_em_dia": ["gripe 2024", "COVID-19"],
        "habitos": {
            "tabagismo": "Ex-fumante (parou há 5 anos)",
            "etilismo": "Social",
            "atividade_fisica": "Caminhada 3x/semana",
            "alimentacao": "Dieta balanceada",
        },
        "comorbidades": ["hipertensão"],
    }

    print(f"\n📋 Paciente: {entrada['nome_paciente']}, {entrada['idade']} anos")
    print(f"   Histórico familiar: {', '.join(entrada['historico_familiar'])}")
    print("\n⏳ Executando fluxo...")

    resultado = fluxo.invoke(entrada)

    exames = resultado.get("exames_devidos", [])
    print(f"\n📊 Exames devidos: {len(exames)}")
    for e in exames:
        print(f"     • {e['exame']} — {e['status']} ({e['intervalo']})")
    agendamentos = resultado.get("agendamentos", [])
    print(f"   Agendamentos criados: {len(agendamentos)}")
    for a in agendamentos:
        print(f"     📅 {a['exame']} — {a['data_sugerida']} ({a['prioridade']})")
    lembretes = resultado.get("lembretes", [])
    print(f"   Lembretes programados: {len(lembretes)}")
    print(f"\n{resultado.get('resumo_final', '')}")
    return resultado


def main():
    print("=" * 60)
    print("  🏥 DEMONSTRAÇÃO — FLUXOS LANGGRAPH SAÚDE DA MULHER")
    print("=" * 60)
    print(f"  Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 60)

    print("\nSelecione o fluxo para demonstrar:\n")
    print("1. Triagem Ginecológica")
    print("2. Detecção de Violência Doméstica")
    print("3. Obstétrico")
    print("4. Prevenção")
    print("5. Executar TODOS os fluxos")
    print("0. Sair")

    try:
        escolha = input("\nOpção: ").strip()
    except KeyboardInterrupt:
        print("\nSaindo...")
        sys.exit(0)

    demos = {
        "1": demo_triagem_ginecologica,
        "2": demo_violencia_domestica,
        "3": demo_obstetrico,
        "4": demo_prevencao,
    }

    if escolha == "5":
        for fn in demos.values():
            try:
                fn()
            except Exception as e:
                print(f"\n❌ Erro: {e}")
    elif escolha in demos:
        try:
            demos[escolha]()
        except Exception as e:
            print(f"\n❌ Erro: {e}")
    elif escolha == "0":
        sys.exit(0)
    else:
        print("⚠️ Opção inválida.")

    print("\n" + "=" * 60)
    print("  ✅ Demonstração finalizada!")
    print("=" * 60)


if __name__ == "__main__":
    main()
