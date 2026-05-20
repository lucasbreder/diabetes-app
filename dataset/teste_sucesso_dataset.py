import json
import os

def validar_dataset_final(file_path):
    print(f"[*] Iniciando auditoria do dataset: {file_path}\n")
    
    if not os.path.exists(file_path):
        print(f"[!] Erro Crítico: O arquivo '{file_path}' não existe.")
        return

    total_registros = 0
    erros_estrutura = 0
    
    # Dicionário para contar quantas amostras temos de cada categoria/fluxo
    contagem_categorias = {
        "Risco Obstétrico/Metabólico": 0,
        "Triagem Ginecológica": 0,
        "Prevenção": 0,
        "Violência Doméstica": 0,
        "Emergência Obstétrica": 0,
        "Saúde Mental da Mulher": 0,
        "Planejamento Familiar e Contracepção": 0,
        "Climatério e Menopausa": 0,
        "Análise de Documentos - Laudo de Mamografia": 0,
        "Procedimentos Especializados - Colposcopia": 0,
        "Outros/Não Identificados": 0
    }
    
    registros_sem_guardrail = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for num_linha, linha in enumerate(f, 1):
            if not linha.strip():
                continue
            
            total_registros += 1
            
            try:
                dados = json.loads(linha)
            except json.JSONDecodeError:
                print(f"[ERRO] Linha {num_linha}: Não é um JSON válido.")
                erros_estrutura += 1
                continue
                
            # 1. Validação de Chaves Obrigatórias
            chaves_obrigatorias = ["instruction", "context", "response"]
            if not all(chave in dados for chave in chaves_obrigatorias):
                print(f"[ERRO] Linha {num_linha}: Faltam chaves obrigatórias. Possui apenas: {list(dados.keys())}")
                erros_estrutura += 1
                continue
                
            # 2. Identificação e Contagem de Categorias
            resposta = dados["response"]
            categoria_encontrada = False
            
            for cat in contagem_categorias.keys():
                if cat in resposta:
                    contagem_categorias[cat] += 1
                    categoria_encontrada = True
                    break
            
            if not categoria_encontrada:
                contagem_categorias["Outros/Não Identificados"] += 1

            # 3. Validação de Cláusula de Segurança (Guardrail)
            termo_seguranca = "DIRETRIZ DE SEGURANÇA"
            if termo_seguranca not in resposta:
                registros_sem_guardrail += 1

    # --- RELATÓRIO FINAL DA AUDITORIA ---
    print("="*50)
    print("📊 RELATÓRIO DE VALIDAÇÃO DO DATASET")
    print("="*50)
    print(f"Total de registros analisados: {total_registros}")
    print(f"Erros de estrutura encontrados: {erros_estrutura}")
    print(f"Registros violando a cláusula de segurança: {registros_sem_guardrail}")
    print("-"*50)
    print("🎯 COBERTURA POR CATEGORIA (Fine-Tuning Target):")
    
    for cat, total in contagem_categorias.items():
        status = "✅" if total > 0 else "❌ ALERTA (Zerar essa categoria reprova o item!)"
        if cat == "Outros/Não Identificados":
            status = "ℹ️"
        print(f" - {cat}: {total} amostras {status}")
        
    print("="*50)
    
    if erros_estrutura == 0 and registros_sem_guardrail == 0 and contagem_categorias["Outros/Não Identificados"] == 0:
        print("🎉 [SUCESSO] O dataset está perfeito e pronto para o Fine-Tuning!")
    else:
        print("⚠️ [ATENÇÃO] Ajuste os pontos apontados acima antes de treinar o modelo.")

if __name__ == "__main__":
    # Executa a validação apontando para o arquivo unificado dentro da pasta dataset/
    validar_dataset_final("dataset_final_llm.jsonl")