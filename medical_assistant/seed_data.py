"""
Dados iniciais: protocolos médicos (FEBRASGO/MS), medicamentos e pacientes demo.
Execute: python -m medical_assistant.seed_data
"""

from __future__ import annotations

from datetime import date

from .database import get_session, init_db
from .models import CicloMenstrual, ExamePreventivo, Medicamento, Paciente, ProntuarioGinecologico, ProtocoloMedico


# ─────────────────────────────────────────────
# Protocolos FEBRASGO / Ministério da Saúde
# ─────────────────────────────────────────────

PROTOCOLOS = [
    {
        "titulo": "Rastreamento do Câncer do Colo do Útero – Papanicolau",
        "categoria": "rastreamento_cancer",
        "palavras_chave": "papanicolau, colo utero, colpocitologia, hpv, cancer cervical, preventivo",
        "fonte": "FEBRASGO / INCA / Ministério da Saúde (2022)",
        "atualizado_em": date(2022, 1, 1),
        "conteudo": """PROTOCOLO: Rastreamento do Câncer do Colo do Útero

INDICAÇÕES POR FAIXA ETÁRIA:
• Mulheres de 25 a 64 anos: realizar Papanicolau a cada 3 anos após dois exames negativos consecutivos com intervalo de 1 ano.
• Mulheres acima de 64 anos: podem encerrar o rastreamento após dois exames negativos nos últimos 5 anos.
• Mulheres imunossuprimidas (HIV+): realizar anualmente.
• Gestantes: não contraindicado; realizar no pré-natal.

ALERTAS:
• Início das relações sexuais antes dos 15 anos é fator de risco aumentado.
• Múltiplos parceiros sexuais aumentam risco.
• Tabagismo é cofator independente.

RESULTADOS ALTERADOS:
• ASC-US: repetir em 6 meses ou realizar teste de HPV.
• LSIL/ASC-H/HSIL: encaminhar para colposcopia.
• Células glandulares atípicas: investigar endocérvice e endométrio.

ENCAMINHAMENTO: Resultado alterado → Colposcopia com biópsia dirigida.""",
    },
    {
        "titulo": "Rastreamento do Câncer de Mama – Mamografia",
        "categoria": "rastreamento_cancer",
        "palavras_chave": "mamografia, mama, cancer mama, rastreamento, ultrassom mama",
        "fonte": "FEBRASGO / INCA / CFM (2023)",
        "atualizado_em": date(2023, 3, 1),
        "conteudo": """PROTOCOLO: Rastreamento do Câncer de Mama

RASTREAMENTO GERAL:
• Mulheres de 40 a 74 anos: mamografia anual pelo SUS (a partir dos 40 anos).
• Mulheres acima de 50 anos: SUS recomenda bianual; FEBRASGO recomenda anual.
• Autoexame das mamas: não substitui mamografia, mas é incentivado como autocuidado.

ALTO RISCO (iniciar rastreamento antes dos 40 anos):
• Histórico familiar de 1º grau com câncer de mama antes dos 50 anos.
• Mutação BRCA1/BRCA2 (conhecida ou suspeita).
• Radioterapia torácica prévia.
• Síndrome de Li-Fraumeni ou Cowden.
→ Para alto risco: mamografia anual + RNM de mama a partir dos 30 anos.

CLASSIFICAÇÃO BI-RADS:
• 0: Inconclusivo – complementar com US ou RNM.
• 1-2: Benigno – manter rastreamento de rotina.
• 3: Provavelmente benigno – controle em 6 meses.
• 4-5: Suspeito de malignidade – biópsia indicada.
• 6: Malignidade conhecida – tratamento.

DENSITOMETRIA ÓSSEA: indicada em mulheres na pós-menopausa (≥65 anos) ou com fatores de risco para osteoporose.""",
    },
    {
        "titulo": "Planejamento Familiar e Métodos Contraceptivos",
        "categoria": "planejamento_familiar",
        "palavras_chave": "anticoncepcional, contraceptivo, planejamento familiar, diu, preservativo, pílula, implante",
        "fonte": "FEBRASGO / OMS / Ministério da Saúde (2022)",
        "atualizado_em": date(2022, 6, 1),
        "conteudo": """PROTOCOLO: Planejamento Familiar e Contracepção

MÉTODOS HORMONAIS COMBINADOS (estrogênio + progestogênio):
• Pílula combinada oral: 91-99% eficácia com uso correto. Contraindicada em tabagistas >35 anos, tromboembolismo, migrânea com aura, HAS.
• Anel vaginal e adesivo transdérmico: mesmas contraindicações.

MÉTODOS SOMENTE PROGESTOGÊNIO:
• Pílula de progestogênio (minipílula): segura durante amamentação.
• Injetável trimestral (DMPA): alta eficácia; pode causar amenorreia prolongada.
• Implante subdérmico: duração 3 anos, eficácia >99%.
• DIU hormonal (levonorgestrel): 3 a 8 anos conforme modelo; reduz sangramento.

DIU DE COBRE:
• Eficácia >99%; não hormonal; pode ser usado como anticoncepção de emergência até 5 dias após relação desprotegida.
• Inserção de preferência no período menstrual ou pós-parto imediato.

ANTICONCEPÇÃO DE EMERGÊNCIA:
• Levonorgestrel 1,5mg: até 72h (eficaz até 120h com menor eficácia).
• Método de Yuzpe: alternativa se levonorgestrel indisponível.

CONTRACEPÇÃO NA ADOLESCÊNCIA: Métodos de longa ação (LARC) são preferíveis. Sempre oferecer dupla proteção (contraceptivo + preservativo contra IST).

ESTERILIZAÇÃO CIRÚRGICA: Lei 9.263/96 – permitida para maiores de 25 anos ou com 2 filhos vivos. Consentimento obrigatório. Período de reflexão de 60 dias.""",
    },
    {
        "titulo": "Pré-natal de Baixo Risco",
        "categoria": "obstetricia",
        "palavras_chave": "pre-natal, gestacao, gravidez, prenatal, consulta obstetrica, gravidez normal",
        "fonte": "Ministério da Saúde – Atenção ao Pré-natal de Baixo Risco (2022)",
        "atualizado_em": date(2022, 1, 1),
        "conteudo": """PROTOCOLO: Pré-natal de Baixo Risco

CALENDÁRIO MÍNIMO DE CONSULTAS (MS):
• 1ª consulta: idealmente até 12 semanas (1º trimestre).
• Total mínimo: 6 consultas (MS) – FEBRASGO recomenda pelo menos 8.
• Distribuição recomendada: mensais até 28 semanas; quinzenais de 28-36 semanas; semanais após 36 semanas.

EXAMES NA 1ª CONSULTA:
• Tipagem sanguínea e fator Rh; hemograma; glicemia de jejum; urina I e urocultura; sorologia: sífilis (VDRL), HIV, hepatite B (HBsAg), hepatite C, toxoplasmose (IgM e IgG), rubéola; TSH; Papanicolau (se necessário).

EXAMES POR TRIMESTRE:
• 1º trimestre (até 13 sem): US morfológico precoce + rastreamento cromossômico (translucência nucal).
• 2º trimestre (18-24 sem): US morfológico do 2º trimestre; TOTG 75g (rastreamento DMG entre 24-28 sem); hemograma; urina I.
• 3º trimestre (28-36 sem): hemograma; sorologias (sífilis, HIV); estreptococo B (entre 35-37 sem); cardiotocografia (>40 sem).

IMUNIZAÇÕES: Influenza, hepatite B (3 doses), dTpa (entre 20-36 semanas).

SINAIS DE ALARME: Sangramento vaginal, hipertensão (PA ≥140/90), cefaleia intensa, epigastralgia, edema súbito, movimentação fetal reduzida → encaminhar imediatamente.""",
    },
    {
        "titulo": "Triagem e Manejo da Violência Doméstica e Sexual",
        "categoria": "violencia_domestica",
        "palavras_chave": "violencia domestica, violencia sexual, vd, maria da penha, agressao, abuso, notificacao compulsoria",
        "fonte": "OPAS / MS / Lei Maria da Penha 11.340/2006 / SINAN",
        "atualizado_em": date(2023, 1, 1),
        "conteudo": """PROTOCOLO: Triagem e Manejo da Violência Doméstica e Sexual

ABORDAGEM INICIAL (ambiente privativo, sem o acompanhante):
• Acolher sem julgamento; garantir privacidade e confidencialidade.
• Perguntar diretamente de forma cuidadosa: "Às vezes, quando as mulheres têm [sintoma], isso está relacionado a problemas em casa. Você se sente segura em casa?"

INSTRUMENTO DE TRIAGEM – WAST (Woman Abuse Screening Tool) adaptado:
As 8 perguntas identificam: violência física, psicológica, sexual e patrimonial.
Pontuação ≥3 indica risco moderado; ≥5 indica alto risco.

NOTIFICAÇÃO COMPULSÓRIA (obrigatória):
• Violência sexual, física, psicológica e negligência são de notificação compulsória pelo profissional de saúde.
• Formulário SINAN (Sistema de Informação de Agravos de Notificação).
• Casos envolvendo crianças/adolescentes: notificar ao Conselho Tutelar imediatamente.

NÍVEIS DE RISCO E CONDUTA:
• BAIXO: Orientação, informação sobre CRAS/CREAS, entrega de material informativo.
• MODERADO: Acionar assistência social, discutir plano de segurança, orientar sobre Delegacia da Mulher.
• ALTO: Acionar equipe multiprofissional, contato com Casa Abrigo, orientar sobre Medida Protetiva de Urgência (MPU).
• CRÍTICO (risco de vida): Acionar SAMU/Bombeiros se necessário, notificação ao Conselho Tutelar se menores envolvidos, suporte emocional emergencial.

PLANO DE SEGURANÇA (orientar a paciente a):
• Ter documentos importantes em local acessível (RG, certidão de nascimento filhos, BO).
• Ter mochila de emergência preparada.
• Ter números de emergência memorizados: 180 (Central da Mulher), 190 (Polícia), 197 (PC).
• Informar pessoa de confiança sobre a situação.

ATENDIMENTO PÓS-VIOLÊNCIA SEXUAL (até 72h):
• Profilaxia IST: azitromicina 1g + ceftriaxona 500mg IM + metronidazol 2g.
• Profilaxia HIV (PEP): iniciar em até 72h, por 28 dias.
• Anticoncepção de emergência: levonorgestrel 1,5mg.
• Coleta de material forense (com consentimento).
• Encaminhar para acompanhamento psicológico.""",
    },
    {
        "titulo": "Síndrome dos Ovários Policísticos (SOP)",
        "categoria": "ginecologia",
        "palavras_chave": "sop, ovarios policisticos, hiperandrogenismo, irregularidade menstrual, infertilidade, acne",
        "fonte": "FEBRASGO / International PCOS Network (2023)",
        "atualizado_em": date(2023, 5, 1),
        "conteudo": """PROTOCOLO: Síndrome dos Ovários Policísticos (SOP)

CRITÉRIOS DE ROTTERDAM (2 de 3 critérios necessários):
1. Irregularidade menstrual (oligo/amenorreia).
2. Hiperandrogenismo clínico (hirsutismo, acne) ou laboratorial (testosterona livre elevada).
3. Morfologia policística nos ovários ao ultrassom (≥20 folículos por ovário ou volume ≥10mL).

DIAGNÓSTICO DIFERENCIAL: Excluir hipotireoidismo (TSH), hiperprolactinemia, hiperplasia adrenal congênita (17-OH-progesterona).

AVALIAÇÃO LABORATORIAL:
• FSH, LH, estradiol (fase folicular); testosterona total e livre; SHBG; DHEA-S; androstenediona; 17-OH-progesterona; prolactina; TSH; glicemia de jejum; TOTG 75g; lipidograma completo.
• Insulina de jejum + HOMA-IR (avaliar resistência insulínica).

MANEJO:
• Mudança de estilo de vida (MEV): perda de 5-10% do peso melhora significativamente ciclos e androgenismo.
• Regularização menstrual: ACO combinados de primeira linha (progestogênio antiandrogênico preferível: ciproterona, drospirenona, dienogesta).
• Hiperandrogenismo: espironolactona 100-200mg/dia ou ACO antiandrogênico.
• Resistência insulínica/diabetes: metformina 1500-2000mg/dia.
• Infertilidade: indução ovulatória com letrozol ou citrato de clomifeno; referência para reprodução assistida se falha.

RASTREAMENTO ASSOCIADO: Monitorar PA, glicemia, perfil lipídico, endométrio (risco de hiperplasia).""",
    },
    {
        "titulo": "Endometriose – Diagnóstico e Manejo",
        "categoria": "ginecologia",
        "palavras_chave": "endometriose, dismenorreia, dor pelvica, infertilidade, endometrioma, laparoscopia",
        "fonte": "FEBRASGO / ESHRE (2022)",
        "atualizado_em": date(2022, 9, 1),
        "conteudo": """PROTOCOLO: Endometriose

SUSPEITA CLÍNICA:
• Dismenorreia intensa não responsiva a AINEs.
• Dor pélvica crônica (>6 meses).
• Dispareunia (dor na relação sexual).
• Subfertilidade sem causa aparente.
• Sintomas intestinais/urinários cíclicos.

DIAGNÓSTICO:
• Clínico + imagem (US transvaginal com preparo intestinal detecta endometriomas e endometriose profunda).
• Padrão-ouro: laparoscopia diagnóstica com biópsia.
• CA-125: auxilia no monitoramento (não diagnóstico isolado).

TRATAMENTO MEDICAMENTOSO (supressão hormonal):
• 1ª linha: ACOs contínuos ou DIU de levonorgestrel (reduz dismenorreia e progressão).
• 2ª linha: Progestogênios (dienogesta 2mg/dia, AMP, noretisterona).
• 3ª linha: Agonistas GnRH (com add-back therapy) – uso máximo 12 meses por ciclo.
• AINEs: para controle da dor (ibuprofeno, naproxeno).

TRATAMENTO CIRÚRGICO:
• Indicado em: falha clínica, endometriomas >4cm, endometriose profunda com obstrução, infertilidade.
• Laparoscopia excisional (preferível a ablação).
• Cirurgia radical (histerectomia ± ooforectomia) apenas em casos refratários sem desejo de gestação futura.

INFERTILIDADE: Referência para reprodução assistida conforme estadiamento.""",
    },
    {
        "titulo": "Climatério e Menopausa – Manejo",
        "categoria": "ginecologia",
        "palavras_chave": "menopausa, climaterio, fogacho, osteoporose, terapia hormonal, TRH, HRT",
        "fonte": "FEBRASGO / NAMS / IMS (2022)",
        "atualizado_em": date(2022, 11, 1),
        "conteudo": """PROTOCOLO: Climatério e Menopausa

DEFINIÇÃO:
• Menopausa: amenorreia por 12 meses consecutivos sem causa identificável (após exclusão de gravidez e patologia).
• Climatério: período de transição de 2-8 anos ao redor da menopausa.
• Menopausa precoce: antes dos 40 anos (investigar insuficiência ovariana prematura).

SINTOMAS E AVALIAÇÃO:
• Vasomotores: fogachos, sudorese noturna (escala MRS ou Kupperman).
• Geniturinários: atrofia vaginal, dispareunia, incontinência urinária.
• Psicológicos: irritabilidade, depressão, insônia, dificuldade de concentração.
• Avaliação: FSH (>40 UI/L confirma pós-menopausa), estradiol, TSH, lipidograma, glicemia, mamografia, densitometria.

TERAPIA HORMONAL (TH) – INDICAÇÕES:
• Sintomas vasomotores moderados a intensos com impacto na qualidade de vida.
• Atrofia urogenital sintomática.
• Prevenção de osteoporose (quando outros tratamentos contraindicados).
• Iniciar idealmente < 60 anos ou nos primeiros 10 anos pós-menopausa ("janela de oportunidade").

CONTRAINDICAÇÕES ABSOLUTAS DA TH:
• Câncer de mama ou endométrio ativos.
• Tromboembolismo venoso ativo.
• Hepatopatia grave.
• Sangramento vaginal não investigado.

MULHERES SEM HISTERECTOMIA: TH combinada (estrogênio + progestogênio) para proteção endometrial.
MULHERES COM HISTERECTOMIA: Estrogênio isolado.
TRATAMENTO LOCAL (atrofia): Estrogênio tópico vaginal (baixo risco sistêmico).

TERAPIAS NÃO HORMONAIS: ISRS/IRSN para fogachos (paroxetina, venlafaxina); isoflavonas como opção complementar.""",
    },
    {
        "titulo": "Rastreamento e Prevenção de IST/HIV na Saúde da Mulher",
        "categoria": "saude_sexual",
        "palavras_chave": "ist, dst, hiv, sifilis, gonorreia, hpv, herpes, clamydia, prevencao, preservativo",
        "fonte": "Ministério da Saúde – Protocolo Clínico IST (2022)",
        "atualizado_em": date(2022, 7, 1),
        "conteudo": """PROTOCOLO: Rastreamento e Prevenção de IST/HIV

RASTREAMENTO DE ROTINA:
• HIV: testar anualmente em mulheres sexualmente ativas até 65 anos; a cada 3 meses em populações de alto risco.
• Sífilis (VDRL/TPHA): anualmente; no pré-natal (1ª consulta + 3º trimestre).
• Hepatite B (HBsAg): triagem única em adultos não vacinados + vacinação.
• HPV: vacinação em meninas 9-14 anos (SUS); disponível para mulheres até 45 anos.
• Clamídia e gonorreia: rastrear em menores de 25 anos sexualmente ativas ou com múltiplos parceiros.

RASTREAMENTO EM GESTANTES:
• HIV + sífilis na 1ª consulta e repetir no 3º trimestre.
• Estreptococo B (S. agalactiae): swab vaginal/retal entre 35-37 semanas.
• Hepatite B: testar e vacinar se negativo.

PREVENÇÃO:
• Dupla proteção: contraceptivo + preservativo masculino ou feminino.
• PrEP (profilaxia pré-exposição ao HIV): indicada em populações de risco; tenofovir/emtricitabina.
• Vacina HPV: duas doses (9-14 anos) ou três doses (≥15 anos).

MANEJO BÁSICO:
• Corrimento vaginal: investigar candidíase, vaginose bacteriana, tricomoníase.
• Úlceras genitais: investigar herpes, sífilis, cancro mole.
• Sempre tratar parceiro(s) para IST curáveis.
• Notificação compulsória: HIV, sífilis, hepatite B/C, gonorreia.""",
    },
    {
        "titulo": "Saúde Mental da Mulher – Depressão e Ansiedade",
        "categoria": "saude_mental",
        "palavras_chave": "depressao, ansiedade, transtorno mental, saude mental, dppp, depressao pos parto, ideacao suicida",
        "fonte": "CFM / FEBRASGO / MS (2022)",
        "atualizado_em": date(2022, 4, 1),
        "conteudo": """PROTOCOLO: Saúde Mental da Mulher

TRIAGEM DE DEPRESSÃO:
• PHQ-2 como triagem: "Nas últimas 2 semanas, você se sentiu deprimida/triste/sem esperança? Teve pouco interesse ou prazer em fazer as coisas?"
• PHQ-9 positivo (≥10): encaminhar para avaliação psicológica/psiquiátrica.
• Edinburgh Postnatal Depression Scale (EPDS): triagem para depressão pós-parto (aplicar 1 semana e 1 mês pós-parto).

DEPRESSÃO PÓS-PARTO (DPP):
• Prevalência: 10-20% das puérperas.
• Diferenciar de "baby blues" (resolve em 2 semanas sem tratamento).
• DPP: sintomas além de 2 semanas; afeta cuidado do bebê.
• Tratamento: psicoterapia (TCC), antidepressivos seguros na amamentação (sertralina, paroxetina).

PSICOSE PUERPERAL: Emergência psiquiátrica. Alucinações, delírios, comportamento desorganizado nos primeiros 30 dias pós-parto → hospitalização imediata.

TRANSTORNO DISFÓRICO PRÉ-MENSTRUAL (TDPM):
• Sintomas graves na fase lútea (humor, irritabilidade, dor) que prejudicam funcionamento.
• Tratamento: ISRS contínuo ou na fase lútea; contraceptivos (drospirenona); modificações de estilo de vida.

IDEAÇÃO SUICIDA: Aplicar Columbia Suicide Severity Rating Scale (C-SSRS). Risco imediato → acionar serviço de emergência psiquiátrica (CAPS, UPA, SAMU).

SUBSTÂNCIAS: Rastrear uso de álcool (AUDIT), tabaco e outras substâncias em todas as consultas. Gestantes: qualquer uso é de alto risco.""",
    },
]

# ─────────────────────────────────────────────
# Medicamentos da Saúde da Mulher
# ─────────────────────────────────────────────

MEDICAMENTOS = [
    {
        "nome_comercial": "Diane 35",
        "principio_ativo": "Acetato de ciproterona 2mg + etinilestradiol 35mcg",
        "categoria": "anticoncepcional_oral_combinado",
        "indicacoes": "Anticoncepcão, hirsutismo, acne androgênica, SOP",
        "contraindicacoes": "Tromboembolismo, migrânea com aura, tabagismo >35 anos, HAS grave, hepatopatia",
        "seguro_gestacao": False,
        "seguro_amamentacao": False,
        "interacoes_importantes": "Rifampicina, fenitoína, carbamazepina (redução eficácia)",
        "observacoes": "Contraindicado exclusivamente como contraceptivo; indicar quando há hiperandrogenismo associado",
    },
    {
        "nome_comercial": "Mirena",
        "principio_ativo": "Levonorgestrel 52mg (DIU hormonal)",
        "categoria": "contraceptivo_intrauterino_hormonal",
        "indicacoes": "Anticoncepção (até 8 anos), menorragia, dismenorreia, proteção endometrial na TRH",
        "contraindicacoes": "Gravidez, infecção pélvica ativa, anomalia uterina, câncer genital",
        "seguro_gestacao": False,
        "seguro_amamentacao": True,
        "interacoes_importantes": "Imunossupressores podem alterar eficácia em casos raros",
        "observacoes": "Pode causar amenorreia em até 50% das usuárias após 1 ano; seguro na amamentação a partir de 6 semanas pós-parto",
    },
    {
        "nome_comercial": "Depo-Provera",
        "principio_ativo": "Acetato de medroxiprogesterona 150mg/mL (injetável)",
        "categoria": "contraceptivo_injetavel",
        "indicacoes": "Anticoncepção, endometriose, amenorreia terapêutica",
        "contraindicacoes": "Câncer de mama, sangramento vaginal não diagnosticado, gravidez",
        "seguro_gestacao": False,
        "seguro_amamentacao": True,
        "interacoes_importantes": "Pode reduzir densidade óssea com uso prolongado",
        "observacoes": "Injeção a cada 3 meses. Fertibilidade pode demorar a retornar (até 12-18 meses). Ideal para quem não tolera estrogênio",
    },
    {
        "nome_comercial": "Plan B / Postinor",
        "principio_ativo": "Levonorgestrel 1,5mg",
        "categoria": "anticoncepcao_emergencia",
        "indicacoes": "Anticoncepção de emergência (até 72h; eficácia reduzida até 120h)",
        "contraindicacoes": "Gestação confirmada (não eficaz, não teratogênico)",
        "seguro_gestacao": False,
        "seguro_amamentacao": True,
        "interacoes_importantes": "Rifampicina e antiepilépticos reduzem eficácia",
        "observacoes": "Quanto mais cedo tomado, maior eficácia. Não protege contra IST. Não é abortivo.",
    },
    {
        "nome_comercial": "Metformina",
        "principio_ativo": "Cloridrato de metformina",
        "categoria": "antidiabetico_biguanida",
        "indicacoes": "Diabetes tipo 2, pré-diabetes, resistência insulínica na SOP",
        "contraindicacoes": "Insuficiência renal (TFG<30), insuficiência hepática, alcoolismo, uso de contraste iodado",
        "seguro_gestacao": True,
        "seguro_amamentacao": True,
        "interacoes_importantes": "Contraste iodado (suspender 48h antes); álcool (risco de acidose lática)",
        "observacoes": "Iniciar com dose baixa (500mg) e aumentar gradualmente para minimizar efeitos GI. Monitorar vitamina B12 com uso prolongado.",
    },
    {
        "nome_comercial": "Dienogeste (Visanne)",
        "principio_ativo": "Dienogeste 2mg",
        "categoria": "progestogenio",
        "indicacoes": "Endometriose (tratamento de manutenção), dismenorreia",
        "contraindicacoes": "Tromboembolismo, doença hepática grave, câncer hormônio-dependente",
        "seguro_gestacao": False,
        "seguro_amamentacao": False,
        "interacoes_importantes": "Indutores do CYP3A4 (rifampicina, fenitoína) reduzem eficácia",
        "observacoes": "Não é contraceptivo; recomenda-se usar método contraceptivo adicional. Pode causar irregularidade menstrual nos primeiros meses.",
    },
    {
        "nome_comercial": "Sertralina",
        "principio_ativo": "Cloridrato de sertralina",
        "categoria": "antidepressivo_isrs",
        "indicacoes": "Depressão, transtorno de ansiedade, TDPM, depressão pós-parto, TEPT",
        "contraindicacoes": "Uso concomitante de IMAO, síndrome serotoninérgica",
        "seguro_gestacao": True,
        "seguro_amamentacao": True,
        "interacoes_importantes": "IMAO (contraindicado), tramadol, linezolida, anticoagulantes",
        "observacoes": "Antidepressivo mais estudado na gestação e lactação; considerado primeira escolha nestes contextos. Iniciar com 25-50mg/dia.",
    },
    {
        "nome_comercial": "Estradiol (Estriol vaginal / Ovestin)",
        "principio_ativo": "Estriol 1mg/g (creme vaginal)",
        "categoria": "estrogeno_topico",
        "indicacoes": "Atrofia urogenital, dispareunia, vaginite atrófica pós-menopausa",
        "contraindicacoes": "Câncer de mama ativo (relativo), sangramento vaginal não investigado",
        "seguro_gestacao": False,
        "seguro_amamentacao": False,
        "interacoes_importantes": "Absorção sistêmica mínima; raramente interage com outros medicamentos",
        "observacoes": "Baixa absorção sistêmica; pode ser usado mesmo em sobreviventes de câncer de mama em alguns casos (avaliar com oncologista). Melhora significativamente a qualidade de vida.",
    },
    {
        "nome_comercial": "Yasmin / Yaz",
        "principio_ativo": "Drospirenona 3mg + etinilestradiol 20-30mcg",
        "categoria": "anticoncepcional_oral_combinado",
        "indicacoes": "Anticoncepção, acne moderada, TDPM (Yaz), retenção hídrica associada ao ciclo, SOP com componente androgênico",
        "contraindicacoes": "Tromboembolismo (TVP/TEP), trombofilia, migrânea com aura, tabagismo >35 anos, HAS grave, hepatopatia, câncer hormônio-dependente, insuficiência renal/adrenal",
        "seguro_gestacao": False,
        "seguro_amamentacao": False,
        "interacoes_importantes": "Indutores enzimáticos (rifampicina, fenitoína, carbamazepina, erva-de-São-João) reduzem eficácia; cuidado com poupadores de potássio (drospirenona tem ação antimineralocorticoide)",
        "observacoes": "Drospirenona é análogo da espironolactona — efeito antiandrogênico e antimineralocorticoide. Yaz tem regime 24/4 (menos sintomas de privação). Risco trombótico ligeiramente superior aos COCs com levonorgestrel.",
    },
    {
        "nome_comercial": "Cerazette / Juliet",
        "principio_ativo": "Desogestrel 75mcg (minipílula)",
        "categoria": "anticoncepcional_progestogenio_isolado",
        "indicacoes": "Anticoncepção durante lactação, mulheres com contraindicação a estrogênio (tabagistas >35 anos, enxaqueca com aura, pós-parto imediato, TVP prévia)",
        "contraindicacoes": "Câncer de mama atual, hepatopatia grave, sangramento vaginal não diagnosticado, gravidez confirmada",
        "seguro_gestacao": False,
        "seguro_amamentacao": True,
        "interacoes_importantes": "Indutores do CYP3A4 (rifampicina, fenitoína, carbamazepina, modafinila) reduzem eficácia; antirretrovirais podem alterar metabolismo",
        "observacoes": "Inibe ovulação (diferente da minipílula clássica de noretisterona). Janela de tomada de 12h. Pode iniciar imediatamente após o parto. Padrão de sangramento irregular é comum nos primeiros 3 meses.",
    },
    {
        "nome_comercial": "Implanon NXT",
        "principio_ativo": "Etonogestrel 68mg (implante subdérmico)",
        "categoria": "contraceptivo_implante_subdermico",
        "indicacoes": "Anticoncepção de longa duração (LARC) por 3 anos, mulheres com contraindicação a estrogênio, adolescentes (recomendação FEBRASGO/OMS)",
        "contraindicacoes": "Gravidez, câncer de mama atual, hepatopatia grave, sangramento vaginal não diagnosticado, TVP/TEP ativa",
        "seguro_gestacao": False,
        "seguro_amamentacao": True,
        "interacoes_importantes": "Rifampicina, anticonvulsivantes (fenitoína, carbamazepina, topiramato), erva-de-São-João — reduzem eficácia significativamente",
        "observacoes": "Eficácia >99% (índice de Pearl 0,05). Inserção subdérmica no braço não-dominante por profissional treinado. Fertilidade retorna em poucos dias após remoção. Sangramento irregular é o efeito colateral mais comum e principal causa de descontinuação.",
    },
    {
        "nome_comercial": "Ácido Fólico 5mg (Folacin / Endofolin)",
        "principio_ativo": "Ácido fólico 5mg (acido folico, folato, vitamina B9, folacina)",
        "categoria": "suplemento_pre_concepcional",
        "indicacoes": "Prevenção de defeitos do tubo neural (espinha bífida, anencefalia), suplementação pré-concepcional (3 meses antes da gestação até 12 semanas), anemia megaloblástica, uso de antiepilépticos",
        "contraindicacoes": "Anemia perniciosa não tratada (mascara deficiência de B12); hipersensibilidade ao princípio ativo",
        "seguro_gestacao": True,
        "seguro_amamentacao": True,
        "interacoes_importantes": "Metotrexato (antagonista), fenitoína (reduz níveis séricos do anticonvulsivante), sulfassalazina, trimetoprima",
        "observacoes": "Dose padrão pré-concepcional: 0,4-0,8mg/dia. Dose 5mg/dia para mulheres com filho prévio com DTN, uso de antiepilépticos, diabetes, obesidade ou anemia falciforme. Iniciar pelo menos 1 mês antes da concepção planejada.",
    },
]

# ─────────────────────────────────────────────
# Pacientes Demo
# ─────────────────────────────────────────────

PACIENTES_DEMO = [
    {
        "nome": "Ana Clara Ferreira",
        "data_nascimento": date(1985, 3, 15),
        "cpf": "111.222.333-44",
        "telefone": "(11) 99988-7766",
        "prontuario": {
            "data_consulta": date(2024, 11, 10),
            "queixas": "Irregularidade menstrual há 6 meses, dor pélvica crônica, acne em adulta",
            "gestacoes": 2, "partos_normais": 1, "partos_cesareos": 1, "abortos": 0,
            "metodo_contraceptivo": "DIU hormonal (Mirena)",
            "ultima_menstruacao": date(2024, 10, 20),
            "historico_dst": "Nenhum relatado",
            "alergias": "Dipirona",
            "medicamentos_uso": "Mirena (DIU)",
            "observacoes": "Suspeita de SOP. Solicitar hormônios.",
            "medico_responsavel": "Dra. Paula Mendes",
        },
        "exames": [
            {
                "tipo_exame": "papanicolau",
                "data_realizacao": date(2022, 3, 10),
                "resultado": "NILM (Negativo para lesão intraepitelial)",
                "resultado_alterado": False,
                "proximo_previsto": date(2025, 3, 10),
                "laboratorio": "Lab Central",
                "medico_solicitante": "Dra. Paula Mendes",
            },
            {
                "tipo_exame": "ultrassom_pelvico",
                "data_realizacao": date(2024, 10, 15),
                "resultado": "Ovários com morfologia policística bilateralmente (>20 folículos/ovário). Endométrio 8mm, homogêneo.",
                "resultado_alterado": True,
                "proximo_previsto": date(2025, 4, 15),
                "laboratorio": "Clínica Imagem",
                "medico_solicitante": "Dra. Paula Mendes",
            },
        ],
        "ciclos": [
            {"data_inicio": date(2024, 10, 20), "data_fim": date(2024, 10, 26), "duracao_dias": 6, "intensidade": "intensa", "dor_escala": 7, "sintomas_associados": "cólica intensa, cefaleia"},
            {"data_inicio": date(2024, 8, 1), "duracao_dias": None, "intensidade": "leve", "dor_escala": 3, "sintomas_associados": "leve cólica"},
        ],
    },
    {
        "nome": "Beatriz Santos Lima",
        "data_nascimento": date(1972, 8, 22),
        "cpf": "222.333.444-55",
        "telefone": "(21) 98877-6655",
        "prontuario": {
            "data_consulta": date(2025, 1, 20),
            "queixas": "Fogachos frequentes, insônia, ressecamento vaginal, última menstruação há 18 meses",
            "gestacoes": 3, "partos_normais": 2, "partos_cesareos": 1, "abortos": 0,
            "metodo_contraceptivo": "Nenhum (pós-menopausa)",
            "ultima_menstruacao": date(2023, 7, 1),
            "historico_dst": "Nenhum",
            "alergias": "Penicilina",
            "medicamentos_uso": "Anlodipina 5mg, Losartana 50mg",
            "observacoes": "Menopausa confirmada (FSH 68 UI/L). Avaliar TH.",
            "medico_responsavel": "Dra. Camila Rocha",
        },
        "exames": [
            {
                "tipo_exame": "mamografia",
                "data_realizacao": date(2023, 9, 5),
                "resultado": "BI-RADS 2 – Achados benignos. Sem nódulos suspeitos.",
                "resultado_alterado": False,
                "proximo_previsto": date(2024, 9, 5),
                "laboratorio": "Instituto de Diagnóstico",
                "medico_solicitante": "Dra. Camila Rocha",
            },
            {
                "tipo_exame": "papanicolau",
                "data_realizacao": date(2022, 11, 12),
                "resultado": "NILM",
                "resultado_alterado": False,
                "proximo_previsto": date(2025, 11, 12),
                "laboratorio": "Lab Central",
                "medico_solicitante": "Dra. Camila Rocha",
            },
            {
                "tipo_exame": "densitometria",
                "data_realizacao": date(2024, 2, 10),
                "resultado": "Osteopenia em coluna lombar (T-score: -1.8). Quadril normal.",
                "resultado_alterado": True,
                "proximo_previsto": date(2026, 2, 10),
                "laboratorio": "Clínica Ossos",
                "medico_solicitante": "Dra. Camila Rocha",
            },
        ],
        "ciclos": [],
    },
    {
        "nome": "Carla Oliveira Nascimento",
        "data_nascimento": date(1998, 12, 5),
        "cpf": "333.444.555-66",
        "telefone": "(31) 97766-5544",
        "prontuario": {
            "data_consulta": date(2025, 3, 8),
            "queixas": "Dor pélvica intensa, dispareunia, menstruação muito dolorosa desde a adolescência",
            "gestacoes": 0, "partos_normais": 0, "partos_cesareos": 0, "abortos": 0,
            "metodo_contraceptivo": "Anticoncepcional oral (Yasmin)",
            "ultima_menstruacao": date(2025, 2, 28),
            "historico_dst": "HPV detectado (2023) – tratado",
            "alergias": "Nenhuma conhecida",
            "medicamentos_uso": "Yasmin, Ibuprofeno (quando necessário)",
            "observacoes": "Forte suspeita de endometriose. US sugestivo. Considerar laparoscopia diagnóstica.",
            "medico_responsavel": "Dr. Roberto Alves",
        },
        "exames": [
            {
                "tipo_exame": "papanicolau",
                "data_realizacao": date(2023, 6, 15),
                "resultado": "ASC-US com positividade para HPV de alto risco (HPV 16/18)",
                "resultado_alterado": True,
                "proximo_previsto": date(2024, 6, 15),
                "laboratorio": "Lab Diagnosta",
                "medico_solicitante": "Dr. Roberto Alves",
            },
            {
                "tipo_exame": "ultrassom_pelvico",
                "data_realizacao": date(2025, 3, 1),
                "resultado": "Endometrioma em ovário direito (2,8cm). Útero com contornos irregulares. Endometriose superficial suspeita.",
                "resultado_alterado": True,
                "proximo_previsto": date(2025, 9, 1),
                "laboratorio": "Clínica Imagem Premium",
                "medico_solicitante": "Dr. Roberto Alves",
            },
        ],
        "ciclos": [
            {"data_inicio": date(2025, 2, 28), "data_fim": date(2025, 3, 6), "duracao_dias": 7, "intensidade": "muito_intensa", "dor_escala": 9, "sintomas_associados": "dismenorreia grave, náusea, impossibilidade de trabalhar"},
            {"data_inicio": date(2025, 1, 29), "data_fim": date(2025, 2, 5), "duracao_dias": 8, "intensidade": "muito_intensa", "dor_escala": 9, "sintomas_associados": "dor incapacitante"},
            {"data_inicio": date(2024, 12, 30), "data_fim": date(2025, 1, 7), "duracao_dias": 9, "intensidade": "muito_intensa", "dor_escala": 8, "sintomas_associados": "sangramento intenso, cólica severa"},
        ],
    },
]


# ─────────────────────────────────────────────
# Função de Seed
# ─────────────────────────────────────────────

def _sincronizar_medicamentos(session) -> tuple[int, int]:
    """Insere medicamentos novos e atualiza os existentes (idempotente por nome_comercial).
    Retorna (novos, atualizados)."""
    indice = {
        m.nome_comercial: m
        for m in session.query(Medicamento).all()
    }
    novos = 0
    atualizados = 0
    for m in MEDICAMENTOS:
        existente = indice.get(m["nome_comercial"])
        if existente is None:
            session.add(Medicamento(**m))
            novos += 1
            continue
        mudou = False
        for campo, valor in m.items():
            if getattr(existente, campo) != valor:
                setattr(existente, campo, valor)
                mudou = True
        if mudou:
            atualizados += 1
    return novos, atualizados


def popular_banco() -> None:
    """Popula o banco com protocolos, medicamentos e pacientes demo.

    O seed de medicamentos é idempotente: novos itens adicionados ao MEDICAMENTOS
    são inseridos mesmo em bancos já populados.
    """
    init_db()

    with get_session() as s:
        ja_populado = s.query(ProtocoloMedico).count() > 0

        if ja_populado:
            novos, atualizados = _sincronizar_medicamentos(s)
            if novos or atualizados:
                print(
                    f"Banco já populado: {novos} novo(s) e {atualizados} atualizado(s) medicamento(s)."
                )
            else:
                print("Banco já populado. Pulando seed.")
            return

        for p in PROTOCOLOS:
            s.add(ProtocoloMedico(**p))

        _sincronizar_medicamentos(s)

        for pd in PACIENTES_DEMO:
            pac = Paciente(
                nome=pd["nome"],
                data_nascimento=pd["data_nascimento"],
                cpf=pd["cpf"],
                telefone=pd["telefone"],
            )
            s.add(pac)
            s.flush()

            pront = ProntuarioGinecologico(paciente_id=pac.id, **pd["prontuario"])
            s.add(pront)

            for ex in pd["exames"]:
                s.add(ExamePreventivo(paciente_id=pac.id, **ex))

            for ciclo in pd["ciclos"]:
                s.add(CicloMenstrual(paciente_id=pac.id, **ciclo))

        print(f"Banco populado: {len(PROTOCOLOS)} protocolos, {len(MEDICAMENTOS)} medicamentos, {len(PACIENTES_DEMO)} pacientes demo.")


if __name__ == "__main__":
    popular_banco()
