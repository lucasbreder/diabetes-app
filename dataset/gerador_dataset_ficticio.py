import json
import random

# Listas de sementes para garantir diversidade socioeconômica, regional e etária (Requisito do Desafio)
perfis_pacientes = [
    {"idade": 22, "contexto_social": "Residente em comunidade periférica, dependente exclusivamente do SUS."},
    {"idade": 35, "contexto_social": "Trabalhadora autônoma, reside em área urbana secundária."},
    {"idade": 48, "contexto_social": "Zona rural, dificuldade de acesso regular a centros de média complexidade."},
    {"idade": 62, "contexto_social": "Beneficiária de programa de assistência social, reside em região metropolitana."}
]

# 1. Dados para o Fluxo de Triagem Ginecológica e Prevenção
cenarios_clinicos = [
    {
        "categoria": "Triagem Ginecológica",
        "sintoma": "dor pélvica de forte intensidade associada a corrimento vaginal amarelado com odor fétido e febre baixa.",
        "protocolo": "Protocolo de Triagem Ginecológica - Doença Inflamatória Pélvica (DIP) - FEBRASGO.",
        "conduta": "Alerta de Urgência de Triagem. O quadro clínico sugere fortemente Doença Inflamatória Pélvica (DIP). Conduta: 1. Encaminhamento imediato para avaliação presencial ginecológica no pronto atendimento; 2. Avaliação de critérios para antibioticoterapia hospitalar ou ambulatorial; 3. Solicitação de hemograma e PCR."
    },
    {
        "categoria": "Prevenção",
        "sintoma": "assintomática, mas relata que nunca realizou o exame de Papanicolau e sua última mamografia foi há 4 anos.",
        "protocolo": "Diretrizes para Detecção Precoce do Câncer de Mama e Colo do Útero - INCA / Ministério da Saúde.",
        "conduta": "Alerta de Exames Preventivos em Atraso. Conduta: 1. Direcionamento para coleta de citopatologia oncótica (Papanicolau); 2. Solicitação de Mamografia de rastreamento (indicada bienalmente para a faixa etária de 50 a 69 anos); 3. Orientações sobre autoexame e sinais de alerta."
    }
]

# 2. Dados para o Fluxo de Detecção de Violência Doméstica (Requisito de Extrema Confidencialidade)
cenarios_violencia = [
    {
        "categoria": "Violência Doméstica",
        "sintoma": "múltiplas escoriações em estágios diferentes de cura. Durante a anamnese, mostra-se extremamente ansiosa, evita contato visual e o companheiro insiste em responder todas as perguntas por ela.",
        "protocolo": "Diretrizes de Manejo e Identificação de Violência Doméstica - Ministério da Saúde / Protocolo de Segurança.",
        "conduta": "ALERTA CRÍTICO DE SEGURANÇA. Padrão comportamental e físico altamente suspeito de violência doméstica. Conduta: 1. Acionamento IMEDIATO e silencioso da equipe multidisciplinar (assistente social e psicóloga); 2. Garantir atendimento em sala isolada sem a presença do acompanhante; 3. Registro e documentação segura e criptografada do caso; 4. Orientação sutil sobre canais de ajuda (Disque 180). NOTA: Manter confidencialidade absoluta."
    }
]

# 3. Novos cenários obrigatórios mapeados para cobrir 100% da Fase 3
cenarios_obrigatorios = [
    # --- PROTOCOLOS E EMERGÊNCIAS ---
    {
        "categoria": "Emergência Obstétrica",
        "sintoma": "gestante na 34ª semana relatando cefaleia intensa, visão borrada (escotomas) e pressão arterial aferida em 160/110 mmHg.",
        "protocolo": "Diretrizes de Emergências Obstétricas - Pré-eclâmpsia Grave - FEBRASGO.",
        "conduta": "ALERTA MÁXIMO DE EMERGÊNCIA OBSTÉTRICA. Quadro altamente sugestivo de Pré-eclâmpsia Grave com iminência de eclâmpsia. Conduta: 1. Direcionamento IMEDIATO para o Centro Obstétrico / Pronto Atendimento Hospitalar; 2. Preparação para protocolo de sulfato de magnésio sob supervisão médica para prevenção de convulsões; 3. Monitoramento contínuo da pressão arterial e vitalidade fetal."
    },
    {
        "categoria": "Saúde Mental da Mulher",
        "sintoma": "puérpera (3 semanas pós-parto) manifestando tristeza profunda incapacitante, crises de choro e sentimentos de extrema culpa ou desapego em relação ao recém-nascido.",
        "protocolo": "Manual de Saúde Mental da Mulher - Depressão Pós-Parto (DPP) - Ministério da Saúde / OMS.",
        "conduta": "Raciocínio Clínico: Sinais clínicos compatíveis com Depressão Pós-Parto (DPP), distanciando-se do 'blues' puerperal devido à persistência e gravidade. Conduta: 1. Encaminhamento prioritário para a equipe de Psicologia Perinatal e Psiquiatria; 2. Agendamento de consulta de acolhimento na UBS; 3. Orientação à rede de apoio familiar sobre vigilância constante e suporte nos cuidados neonatais."
    },
    
    # --- PERGUNTAS FREQUENTES (FAQs) ---
    {
        "categoria": "Planejamento Familiar e Contracepção",
        "sintoma": "solicita orientação sobre esquecimento de pílula anticoncepcional combinada há mais de 24 horas e refere dúvida sobre eficácia.",
        "protocolo": "Critérios Médicos de Elegibilidade para Uso de Contraceptivos - OMS.",
        "conduta": "Raciocínio Clínico: Manejo de esquecimento de anticoncepcional oral hormonal. Conduta: 1. Orientar a tomar o comprimido esquecido imediatamente (mesmo que signifique tomar dois no mesmo dia); 2. Utilizar método de barreira (preservativo) pelos próximos 7 dias consecutivos; 3. Fornecer informações sobre a anticoncepção de emergência se houver ocorrido relação desprotegida nas últimas 72h."
    },
    {
        "categoria": "Climatério e Menopausa",
        "sintoma": "mulher de 52 anos relatando fogachos (ondas de calor) intensos, sudorese noturna, insônia e irregularidade menstrual severa.",
        "protocolo": "Consenso Brasileiro de Terapêutica Hormonal na Menopausa - SOBRAC / FEBRASGO.",
        "conduta": "Raciocínio Clínico: Síndrome climatérica com impacto severo na qualidade de vida. Conduta: 1. Triagem e agendamento de consulta ginecológica para avaliação de critérios de elegibilidade para Terapia Hormonal (TH); 2. Solicitação de exames basais: mamografia, perfil lipídico e glicemia de jejum antes de qualquer conduta terapêutica."
    },

    # --- MODELOS DE DOCUMENTOS ESPECIALIZADOS ---
    {
        "categoria": "Análise de Documentos - Laudo de Mamografia",
        "sintoma": "apresenta laudo impresso de Mamografia digital de rastreamento com conclusão classificada como BI-RADS 4 (Achado Suspeito).",
        "protocolo": "Diretrizes de Rastreamento do Câncer de Mama - Sistema BI-RADS - INCA.",
        "conduta": "Alerta de Achado Suspeito em Exame de Imagem. Conduta: 1. Encaminhamento de urgência para o Mastologista; 2. Preparação de guia de encaminhamento para procedimento de biópsia percutânea (Core Biopsy ou agulhamento) para correlação histopatológica; 3. Acolhimento e explicação clara à paciente de que a classificação exige investigação, mas não firma diagnóstico definitivo de malignidade."
    },
    {
        "categoria": "Procedimentos Especializados - Colposcopia",
        "sintoma": "encaminhada após exame citopatológico (Papanicolau) demonstrar lesão intraepitelial escamosa de alto grau (HSIL).",
        "protocolo": "Diretrizes Brasileiras para o Rastreamento do Câncer do Colo do Útero - INCA.",
        "conduta": "Raciocínio Clínico: Necessidade de correlação diagnóstica estruturada. Conduta: 1. Agendamento prioritário para a realização de exame de Colposcopia com avaliação da zona de transformação; 2. Orientação de que, se identificadas áreas acetobrancas ou mosaicos grosseiros, será realizada biópsia dirigida do colo uterino; 3. Esclarecimento de dúvidas sobre o procedimento clínico."
    }
]

def gerar_dataset_sintetico(output_path, num_registros=100):
    registros = []
    
    # Unificamos todos os pools de cenários para garantir a cobertura completa no sorteio
    todos_os_cenarios = cenarios_clinicos + cenarios_violencia + cenarios_obrigatorios
    
    for _ in range(num_registros):
        perfil = random.choice(perfis_pacientes)
        
        # Sorteia de forma equilibrada de dentro de toda a base de cenários obrigatórios
        cenario = random.choice(todos_os_cenarios)
            
        # Constrói a instrução (Prontuário Anonimizado)
        instrucao = (
            f"Paciente de {perfil['idade']} anos. Perfil: {perfil['contexto_social']} "
            f"Apresenta o seguinte quadro na recepção: {cenario['sintoma']}"
        )
        
        contexto = cenario['protocolo']
        
        # Garante as cláusulas de barreiras e limites de atuação exigidos no projeto
        resposta = (
            f"Raciocínio Clínico Baseado em Evidências ({cenario['categoria']}): "
            f"{cenario['conduta']} "
            "DIRETRIZ DE SEGURANÇA: Este sistema atua estritamente como suporte à decisão clínica de triagem. "
            "É proibida a prescrição automatizada de medicamentos ou o diagnóstico definitivo sem a validação física e assinatura de um médico especialista."
        )
        
        registros.append({
            "instruction": instrucao,
            "context": contexto,
            "response": resposta
        })
        
    # Escreve no formato JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for reg in registros:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
            
    print(f"[+] Dataset sintético gerado com sucesso! Total de {len(registros)} novos cenários em '{output_path}'.")

if __name__ == "__main__":
    # Gerando um volume robusto para cobrir bem todos os novos tópicos injetados
    gerar_dataset_sintetico("saude_mulher_sintetico.jsonl", num_registros=200)