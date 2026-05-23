"""
Script para adicionar pacientes, prontuários e exames ao banco do Assistente Médico.
Edite os dados abaixo e rode: python3 adicionar_paciente.py
"""

from datetime import date
from medical_assistant.database import (
    criar_paciente,
    criar_prontuario,
    criar_exame,
    listar_pacientes,
)

# ─────────────────────────────────────────────
# 1. DADOS DA PACIENTE — edite aqui
# ─────────────────────────────────────────────
NOME = "Maria da Silva"
DATA_NASCIMENTO = date(1990, 3, 15)
CPF = None        
TELEFONE = None  

# ─────────────────────────────────────────────
# 2. PRONTUÁRIO — edite aqui
# ─────────────────────────────────────────────
PRONTUARIO = dict(
    data_consulta=date.today(),
    queixas="Dor pélvica, fluxo intenso",
    gestacoes=1,
    partos_normais=1,
    partos_cesareos=0,
    abortos=0,
    metodo_contraceptivo="Anticoncepcional oral",
    ultima_menstruacao=date(2026, 5, 1),
    historico_dst="Nenhum",
    alergias="Nenhuma",
    medicamentos_uso="Yasmin",
    observacoes="Investigar endometriose.",
    medico_responsavel="Dra. Paula Mendes",
)

# ─────────────────────────────────────────────
# 3. EXAMES — adicione quantos precisar 
# ─────────────────────────────────────────────
EXAMES = [
    dict(
        tipo_exame="papanicolau",
        data_realizacao=date(2025, 6, 10),
        resultado="NILM (Negativo para lesão intraepitelial)",
        resultado_alterado=False,
        proximo_previsto=date(2026, 6, 10),
        laboratorio="Lab Central",
        medico_solicitante="Dra. Paula Mendes",
    ),
    # Adicione mais blocos dict(...) aqui se necessário
]

# ─────────────────────────────────────────────
# Execução — não precisa editar abaixo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    p = criar_paciente(NOME, DATA_NASCIMENTO, CPF, TELEFONE)
    print(f"✅ Paciente criada — ID: {p.id} | Nome: {p.nome}")

    pr = criar_prontuario(paciente_id=p.id, **PRONTUARIO)
    print(f"✅ Prontuário registrado — ID: {pr.id} | Consulta: {pr.data_consulta}")

    for ex in EXAMES:
        e = criar_exame(paciente_id=p.id, **ex)
        print(f"✅ Exame registrado — ID: {e.id} | Tipo: {e.tipo_exame}")

    print("\n📋 Pacientes no banco:")
    for pac in listar_pacientes():
        print(f"  [{pac.id}] {pac.nome}")
