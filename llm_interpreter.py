import ollama
from typing import Optional


def interpretar_resultado(dados_paciente: dict, predicao: int, probabilidade: float) -> Optional[str]:
    """
    Usa o LLama (via Ollama) para gerar uma interpretação em linguagem natural
    do resultado do diagnóstico de diabetes.
    
    Args:
        dados_paciente: Dicionário com os dados clínicos do paciente.
        predicao: 0 (negativo) ou 1 (positivo para diabetes).
        probabilidade: Percentual de certeza do modelo (0.0 a 1.0).
    
    Returns:
        Texto interpretativo gerado pela LLM, ou None em caso de falha.
    """
    resultado_texto = "POSITIVO para diabetes" if predicao == 1 else "NEGATIVO (saudável)"
    certeza = f"{probabilidade:.1%}"

    prompt = f"""Você é um assistente médico especializado em diabetes. 
Analise os dados clínicos abaixo e a predição do modelo de IA, e forneça uma interpretação clara e acessível para o paciente.

**Dados Clínicos do Paciente:**
- Gravidezes: {dados_paciente.get('Pregnancies', 'N/A')}
- Glicose: {dados_paciente.get('Glucose', 'N/A')} mg/dL
- Pressão Sanguínea: {dados_paciente.get('BloodPressure', 'N/A')} mm Hg
- Espessura da Pele: {dados_paciente.get('SkinThickness', 'N/A')} mm
- Insulina: {dados_paciente.get('Insulin', 'N/A')} mu U/ml
- IMC: {dados_paciente.get('BMI', 'N/A')}
- Histórico Familiar (Pedigree): {dados_paciente.get('DiabetesPedigreeFunction', 'N/A')}
- Idade: {dados_paciente.get('Age', 'N/A')} anos

**Resultado do Modelo de IA:** {resultado_texto}
**Certeza do Modelo:** {certeza}

Por favor, forneça:
1. Uma explicação do resultado em linguagem simples
2. Quais valores clínicos chamam mais atenção (acima ou abaixo dos limites normais)
3. Recomendações gerais de saúde

IMPORTANTE: Termine sempre com o aviso de que esta análise é gerada por IA e NÃO substitui uma consulta médica profissional.
Responda em português brasileiro. Seja conciso (máximo 15 linhas).
"""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content
    except Exception as e:
        print(f"\n⚠️  Não foi possível gerar a interpretação da IA: {e}")
        print("Verifique se o Ollama está rodando: ollama serve")
        return None
