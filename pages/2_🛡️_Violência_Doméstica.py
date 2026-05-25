import streamlit as st
from utils.shared_styles import aplicar_estilos_globais, exibir_disclaimer
from utils.paciente_selector import render_seletor_sidebar
from utils.fluxo_runner import executar_fluxo_com_progresso
from utils.form_reset import render_botao_nova_consulta

st.set_page_config(page_title="Violência Doméstica", page_icon="🛡️", layout="wide")
aplicar_estilos_globais()

st.markdown("""
<div class="main-header">
    <h1>🛡️ Detecção de Violência Doméstica</h1>
    <p>Sinais de alerta • Avaliação de risco • Protocolo de segurança • Seguimento</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# Seletor de paciente (sidebar)
# ========================================
contexto = render_seletor_sidebar()

prefill_nome = contexto.nome if contexto else ""
prefill_idade = contexto.idade if contexto else 28

# ========================================
# Formulário
# ========================================
with st.form("form_violencia"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 Dados da Paciente")
        nome = st.text_input("Nome da paciente", value=prefill_nome, placeholder="Ex: Ana Oliveira")
        idade = st.number_input("Idade", min_value=1, max_value=100, value=prefill_idade)
        historico_atend = st.number_input(
            "Nº de atendimentos anteriores no último ano", min_value=0, max_value=50, value=0,
        )
        acompanhante = st.checkbox("Acompanhante presente na consulta")

    with col2:
        st.markdown("#### 📝 Relato")
        relato = st.text_area(
            "Relato da paciente sobre lesões/queixas",
            placeholder="Descreva o que a paciente relatou...",
            height=120,
        )

    st.markdown("#### ⚠️ Sinais de Alerta Observados")
    sinais = st.multiselect("Sinais identificados pelo profissional", [
        "Lesões em diferentes estágios de cicatrização",
        "Lesões incompatíveis com relato",
        "Parceiro controlador presente",
        "Paciente evita contato visual",
        "Relato inconsistente sobre lesões",
        "Múltiplas visitas ao pronto-socorro",
        "Isolamento social relatado",
        "Medo verbalizado",
        "Tentativa de minimizar lesões",
        "Atraso na busca por atendimento",
        "Lesões durante a gravidez",
        "Comportamento ansioso ou submisso",
        "Sinais de desnutrição",
        "Marcas de contenção",
    ])

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🩹 Lesões Observadas")
        lesoes = st.multiselect("Lesões identificadas no exame", [
            "Equimose em face", "Equimose em braço/mão",
            "Equimose em tronco", "Escoriações", "Fraturas",
            "Queimaduras", "Marcas de mordida",
            "Lesões em região genital", "Hematomas múltiplos",
        ])
    with col4:
        st.markdown("#### 👁️ Comportamento Observado")
        comportamento = st.multiselect("Comportamento durante consulta", [
            "Paciente evita contato visual",
            "Comportamento ansioso ou submisso",
            "Choro frequente",
            "Respostas monossilábicas",
            "Olha para acompanhante antes de responder",
            "Treme ao ser examinada",
            "Relutância em despir-se para exame",
        ])

    submitted = st.form_submit_button("🚀 Avaliar Risco", use_container_width=True, type="primary")

# ========================================
# Execução do fluxo
# ========================================
if submitted:
    if not nome or (not sinais and not lesoes and not comportamento):
        st.warning("⚠️ Preencha o nome e selecione ao menos um sinal, lesão ou comportamento.")
        st.stop()

    from flows.violencia_domestica import criar_fluxo_violencia_domestica

    paciente_id_str = (
        f"PAC-{contexto.paciente_id:04d}" if contexto else f"PAC-{hash(nome) % 10000:04d}"
    )

    entrada = {
        "paciente_id": paciente_id_str,
        "nome_paciente": nome,
        "idade": idade,
        "sinais_alerta": sinais,
        "relato_paciente": relato or "Não disponível",
        "lesoes_observadas": lesoes,
        "historico_atendimentos": historico_atend,
        "acompanhante_presente": acompanhante,
        "comportamento_observado": comportamento,
    }

    fluxo = criar_fluxo_violencia_domestica()
    resultado = executar_fluxo_com_progresso(
        fluxo,
        entrada,
        titulo="⏳ Avaliando risco de violência doméstica...",
        titulo_final="✅ Avaliação concluída",
        rotulos_nos={
            "identificar_sinais_alerta": "Identificando sinais de alerta",
            "avaliar_risco": "Avaliando nível de risco",
            "definir_protocolo_seguranca": "Definindo protocolo de segurança",
            "acionar_equipe": "Acionando equipe especializada",
            "documentar_caso": "Documentando caso (sigiloso)",
            "planejar_seguimento": "Planejando seguimento",
        },
    )

    # ========================================
    # Resultados
    # ========================================
    st.divider()
    col_titulo, col_acao = st.columns([3, 1])
    with col_titulo:
        st.markdown("## 📊 Avaliação de Risco")
    with col_acao:
        render_botao_nova_consulta(chave_botao="btn_nova_avaliacao_vd")

    nivel = resultado.get("nivel_risco", "baixo")
    avaliacao = resultado.get("avaliacao_risco", {})
    score = avaliacao.get("score", 0)

    # Card de risco principal
    cores = {
        "critico": ("#7f1d1d", "#fecaca", "🚨 CRÍTICO", "alerta-critico"),
        "alto": ("#991b1b", "#fee2e2", "🔴 ALTO", "urgencia-vermelho"),
        "moderado": ("#92400e", "#fef3c7", "🟡 MODERADO", "urgencia-amarelo"),
        "baixo": ("#166534", "#dcfce7", "🟢 BAIXO", "urgencia-verde"),
    }
    cor_txt, cor_bg, label, classe = cores.get(nivel, cores["baixo"])

    if nivel == "critico":
        st.markdown(f"""
        <div class="alerta-critico" style="text-align:center;">
            <h2 style="color:white;">🚨 RISCO CRÍTICO — AÇÃO IMEDIATA NECESSÁRIA</h2>
            <p style="color:#fecaca;font-size:1.1rem;">Score de risco: {score} | Nível: CRÍTICO</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="{classe}" style="text-align:center;">
            <h3>{label} — Score: {score}</h3>
        </div>
        """, unsafe_allow_html=True)

    # Detalhes
    col_proto, col_equipe = st.columns(2)

    with col_proto:
        st.markdown("### 🔒 Protocolo de Segurança")
        proto = resultado.get("protocolo_seguranca", {})
        st.markdown(f"""
        <div class="info-card">
            <strong>🎯 Ação imediata:</strong> {proto.get('acao_imediata', 'N/A')}<br>
            <strong>🚪 Separar acompanhante:</strong> {'✅ Sim' if proto.get('separar_acompanhante') else '❌ Não'}<br>
            <strong>🏠 Ambiente:</strong> {proto.get('ambiente_seguro', 'N/A')}<br>
            <strong>📞 Contatos:</strong> {', '.join(proto.get('contato_emergencia', []))}<br>
            <strong>⚡ Prioridade:</strong> {proto.get('prioridade', 'N/A')}
        </div>
        """, unsafe_allow_html=True)

    with col_equipe:
        st.markdown("### 👥 Equipe Acionada")
        equipe = resultado.get("equipe_acionada", {})
        st.markdown(f"""
        <div class="info-card">
            <strong>👩‍⚕️ Profissionais:</strong><br>
            {'<br>'.join('• ' + p for p in equipe.get('profissionais', []))}<br><br>
            <strong>🏛️ Órgãos externos:</strong><br>
            {'<br>'.join('• ' + o for o in equipe.get('orgaos_externos', [])) or '—'}<br><br>
            <strong>📋 Notificação compulsória:</strong>
            {'✅ SIM — Obrigatória' if equipe.get('notificacao_compulsoria') else '❌ Não obrigatória'}<br>
            <strong>⏱️ Prazo:</strong> {equipe.get('prazo_acionamento', 'N/A')}
        </div>
        """, unsafe_allow_html=True)

    # Documentação
    st.markdown("### 📄 Documentação do Caso")
    doc = resultado.get("documentacao", {})
    st.info(f"🔐 **Registro sigiloso** — Acesso restrito à equipe autorizada. "
            f"Notificação compulsória: {'Sim' if doc.get('notificacao_compulsoria') else 'Não'}.")

    # Seguimento
    seguimento = resultado.get("plano_seguimento", {})
    st.markdown("### 📋 Plano de Seguimento")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Retorno em", seguimento.get("retorno_em", "—"))
        st.metric("Acomp. Psicológico", "Sim" if seguimento.get("acompanhamento_psicologico") else "Não")
    with col_s2:
        st.metric("Acomp. Social", "Sim" if seguimento.get("acompanhamento_social") else "Não")
        st.markdown("**Rede de apoio:**")
        for r in seguimento.get("rede_apoio", []):
            st.markdown(f"- 📞 {r}")

    # Orientações LLM
    orient = seguimento.get("orientacoes_seguimento", "")
    if orient and "[LLM indisponível" not in orient:
        st.markdown("### 🤖 Orientações da IA")
        st.markdown(f'<div class="llm-card">{orient}</div>', unsafe_allow_html=True)

    exibir_disclaimer()
