import os
import sys

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("========================================")
    print("========================================")
    print("   SISTEMA DE DIAGNÓSTICO DE DIABETES   ")
    print("========================================")
    print("\nComo você deseja executar o projeto?\n")
    print("1. Abrir interface web com Docker (Streamlit + Ollama automaticamente)")
    print("2. Rodar demonstração teste da pipeline localmente no terminal")
    print("3. Sair")
    
    try:
        escolha = input("\nDigite a opção desejada (1, 2 ou 3): ")
    except KeyboardInterrupt:
        print("\nSaindo...")
        sys.exit(0)
        
    if escolha == '1':
        print("\n🐳 Iniciando execução via Docker Compose...")
        print("Isso pode demorar alguns instantes na primeira vez.\n")
        # Roda o docker compose
        retorno = os.system("docker compose up -d --build")
        
        if retorno == 0:
            print("\n✅ Sucesso! A aplicação Streamlit está rodando em plano de fundo.")
            print("🌐 Acesse no seu navegador: http://localhost:8501")
            print("🛑 Para parar o servidor futuramente, execute: docker compose down")
        else:
            print("\n❌ Houve um erro ao iniciar o Docker Compose. Verifique se o Docker Desktop está rodando.")
            
    elif escolha == '2':
        print("\n📥 Verificando e baixando o modelo de IA base (llama3.2:1b). Isso pode demorar na 1ª vez...")
        os.system("ollama pull llama3.2:1b")
        print("\n🚀 Iniciando demonstração da pipeline de Machine Learning...")
        os.system(f"{sys.executable} demo_pipeline.py")
        
    elif escolha == '3':
        print("\nSaindo...")
        sys.exit(0)
    else:
        print("\n⚠️ Opção inválida. Digite 1, 2 ou 3.")

if __name__ == "__main__":
    main()
