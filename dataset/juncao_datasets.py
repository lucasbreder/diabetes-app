import pandas as pd
import json
import os

def converter_linha_csv_para_jsonl(row):
    """Transforma uma linha numérica do Pima CSV em um formato de prompt de texto."""
    gestacoes = int(row['Pregnancies'])
    glicose = row['Glucose']
    pressao = row['BloodPressure']
    imc = row['BMI']
    idade = int(row['Age'])
    resultado = int(row['Outcome'])
    
    instrucao = (
        f"Paciente do sexo feminino, {idade} anos, com histórico de {gestacoes} gestações. "
        f"Apresenta os seguintes parâmetros clínicos na triagem: Glicemia de jejum: {glicose} mg/dL, "
        f"Pressão Arterial: {pressao} mmHg, e Índice de Massa Corporal (IMC): {imc} kg/m²."
    )
    
    contexto = "Protocolo de Triagem e Diretrizes de Risco Obstétrico e Metabólico - FEBRASGO / OMS."
    
    if resultado == 1:
        resposta = (
            "Raciocínio Clínico Baseado em Evidências (Risco Obstétrico/Metabólico): Os níveis glicêmicos e parâmetros "
            "reportados indicam alto risco para distúrbios metabólicos (como Diabetes Mellitus ou Diabetes Gestacional). "
            "Conduta Recomendada: 1. Encaminhamento imediato para avaliação com Ginecologista/Obstetra; "
            "2. Solicitação de Teste Oral de Tolerância à Glicose (TOTG); 3. Orientação nutricional preventiva preliminar. "
            "DIRETRIZ DE SEGURANÇA: Este sistema atua estritamente como suporte à decisão clínica de triagem. "
            "É proibida a prescrição automatizada de medicamentos ou o diagnóstico definitivo sem a validação de um especialista."
        )
    else:
        resposta = (
            "Raciocínio Clínico Baseado em Evidências (Risco Obstétrico/Metabólico): Paciente apresenta parâmetros "
            "glicêmicos e metabólicos dentro da faixa de normalidade até o momento da avaliação. "
            "Conduta Recomendada: Manter o acompanhamento preventivo de rotina ginecológica e obstétrica anual. "
            "DIRETRIZ DE SEGURANÇA: Este sistema atua estritamente como suporte à decisão clínica de triagem. "
            "É proibida a prescrição automatizada de medicamentos ou o diagnóstico definitivo sem a validação de um especialista."
        )
        
    return {"instruction": instrucao, "context": contexto, "response": resposta}

def unificar_datasets():
    csv_path = "diabetes.csv"
    jsonl_sintetico_path = "saude_mulher_sintetico.jsonl"
    output_path = "dataset_final_llm.jsonl"
    
    registros_finais = []
    
    # 1. Processa e converte o arquivo CSV
    if os.path.exists(csv_path):
        print(f"[*] Convertendo dados do arquivo '{csv_path}'...")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            registros_finais.append(converter_linha_csv_para_jsonl(row))
    else:
        print(f"[!] Erro: Arquivo '{csv_path}' não encontrado na pasta local.")
        return

    # 2. Lê e anexa os dados do JSONL sintético
    if os.path.exists(jsonl_sintetico_path):
        print(f"[*] Lendo dados do arquivo '{jsonl_sintetico_path}'...")
        with open(jsonl_sintetico_path, "r", encoding="utf-8") as f:
            for linha in f:
                if linha.strip():
                    registros_finais.append(json.loads(linha))
    else:
        print(f"[!] Erro: Arquivo '{jsonl_sintetico_path}' não encontrado. Rode o gerador antes.")
        return

    # 3. Salva tudo no arquivo unificado final
    with open(output_path, "w", encoding="utf-8") as f:
        for reg in registros_finais:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
            
    print(f"[+] Sucesso! Arquivo unificado '{output_path}' gerado com {len(registros_finais)} registros prontos para o Fine-Tuning.")

if __name__ == "__main__":
    unificar_datasets()