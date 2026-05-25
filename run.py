import os
import socket
import sys


def porta_em_uso(porta):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", porta)) == 0


def main():
    os.system("cls" if os.name == "nt" else "clear")
    print("========================================")
    print("   SISTEMA DE DIAGNÓSTICO DE DIABETES   ")
    print("    + Fluxos de Saúde da Mulher (IA)    ")
    print("========================================")
    print("\nComo você deseja executar o projeto?\n")
    print("1. Abrir interface web com Docker (Streamlit + Ollama)")
    print("2. Rodar demonstração da pipeline localmente (Terminal)")
    print("3. Parar todos os serviços Docker")
    print("4. 🏥 Fluxos LangGraph — Saúde da Mulher")
    print("5. Sair")

    try:
        escolha = input("\nDigite a opção desejada: ")
    except KeyboardInterrupt:
        print("\nSaindo...")
        sys.exit(0)

    if escolha == "1":
        # Verificação preventiva de conflito de porta
        if porta_em_uso(11434):
            print("\n⚠️  Atenção: A porta 11434 já está em uso por outro processo.")
            print(
                "Se o Ollama estiver rodando no seu sistema, pare-o com: 'sudo systemctl stop ollama'"
            )
            input("\nPressione Enter para tentar continuar ou Ctrl+C para sair...")

        print("\n🐳 Iniciando execução via Docker Compose...")
        retorno = os.system("docker compose up -d --build")

        if retorno == 0:
            print("\n✅ Sucesso! A aplicação está subindo.")
            print("⏳ O modelo IA está sendo baixado em segundo plano.")
            print("🌐 Acesse: http://localhost:8501")
            print(
                "\n💡 Dica: Use 'docker logs -f ollama_init' para ver o progresso do download."
            )
        else:
            print(
                "\n❌ Erro ao iniciar Docker. Verifique se o daemon do Docker está ativo."
            )

    elif escolha == "2":
        print("\n📥 Verificando modelo llama3.2:1b no Ollama local...")
        os.system("ollama pull llama3.2:1b")
        print("\n🚀 Iniciando demonstração...")
        os.system(f"{sys.executable} demo_pipeline.py")

    elif escolha == "3":
        print("\n🛑 Parando e removendo containers...")
        os.system("docker compose down")
        print("✅ Tudo limpo!")

    elif escolha == "4":
        print("\n🏥 Iniciando fluxos LangGraph de Saúde da Mulher...")
        os.system(f"{sys.executable} demo_flows.py")

    elif escolha == "5":
        sys.exit(0)
    else:
        print("\n⚠️ Opção inválida.")


if __name__ == "__main__":
    main()
