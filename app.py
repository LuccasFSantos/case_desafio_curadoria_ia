import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuração da página
load_dotenv()
st.set_page_config(
    page_title="Mapeamento Executivo de Talentos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS adaptada para Tema Escuro Executivo (Slate / Dark)
st.markdown("""
<style>
    /* Fundo escuro e tipografia geral */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    .main-title {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        font-weight: 700;
        color: #F8FAFC;
        font-size: 2rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    
    .sub-title {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: #94A3B8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Card executivo escuro com bordas e contraste suaves */
    .executive-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 16px;
        color: #F8FAFC;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }

    /* Botão com destaque em azul corporativo */
    .stButton>button {
        background-color: #2563EB;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: 1px solid #2563EB;
        width: 100%;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        background-color: #1D4ED8;
        border-color: #1D4ED8;
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* Entrada de texto e inputs em modo escuro */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #334155;
        background-color: #1E293B;
        color: #F8FAFC;
    }

    .stTextArea textarea:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 1px #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def carregar_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3})

def elaborar_parecer_executivo(contexto: str, job_description: str) -> str:
    client = genai.Client()
    
    prompt_consultivo = """
Você é um Partner Sênior de Executive Search. Sua tarefa é elaborar um parecer técnico e consultivo sobre a adequação de 3 candidatos mapeados, inicie com um resumo do melhor candidato e porque, sem titulo.

### REGRAS ANTI-ALUCINAÇÃO (ANCORAGEM RÍGIDA):
1. **Exclusividade de Fonte:** Baseie-se EXCLUSIVAMENTE nas informações contidas nos blocos <requisitos> e <candidatos>. É estritamente proibido inferir, deduzir, supor ou extrapolar experiências, competências, formações ou conquistas que não estejam expressamente escritas no texto.
2. **Tratamento de Dados Ausentes:** Se um requisito da vaga não for mencionado no perfil do candidato, declare explicitamente: "Dado não informado no perfil" em vez de assumir que o candidato possui ou não a competência.
3. **Evidência Factual:** Para cada Ponto Forte ou Lacuna apontada, você deve citar o fato exato presente no perfil que sustenta a afirmação.

---

### ESTRUTURA DO PARECER:

Para cada um dos 3 executivos:

1. **Nome e Posição Atual**
2. **Pontos Fortes:** Requisitos cruciais explicitamente atendidos (com comprovem no texto).
3. **Lacunas / Riscos:** Requisitos cruciais não atendidos ou informações omitidas no perfil.
4. **Grau de Aderência (Alta, Média ou Baixa):**
   - **Alta:** Atende explicitamente a todos os requisitos cruciais.
   - **Média:** Atende a parte dos requisitos cruciais e possui lacunas gerenciáveis.
   - **Baixa:** Não atende aos requisitos cruciais centrais ou possui ausência crítica de dados.

Perfis Recuperados da Base:
{context}

Descrição da Vaga (Job Description):
{question}

"""

    conteudo = f"""
{prompt_consultivo}

EXECUTIVOS MAPEADOS NA BASE:
{contexto}

DESAFIO DA POSIÇÃO (JOB DESCRIPTION):
{job_description}

PARECER TÉCNICO DE ADERÊNCIA:
"""

    config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=0.95,
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=conteudo,
        config=config
    )
    return response.text

# Cabeçalho
st.markdown('<div class="main-title">Mapeamento Executivo de Talentos</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Painel de Avaliação Consultiva e Análise de Compatibilidade</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Atalhos")
st.sidebar.markdown("Selecione um perfil:")

opcao_vaga = st.sidebar.selectbox(
    "Job Descriptions para Teste:",
    ["Personalizado", "Tech Lead / CTO (Escala & Dados)", "CFO (Reestruturação & M&A)"]
)

vaga_1_txt = "Startup B2B em fase de expansão busca CTO com atuação em arquitetura de dados e inteligência computacional. Perfil hands-on para estruturação de time técnico do zero e garantia de escalabilidade das soluções."
vaga_2_txt = "Empresa de tecnologia busca CFO para liderar captação de recursos institucionais e conduzir processo de M&A. Exige histórico em gestão financeira de empresas de alto crescimento e relacionamento com fundos de capital de risco."

if opcao_vaga == "Tech Lead / CTO (Escala & Dados)":
    jd_padrao = vaga_1_txt
elif opcao_vaga == "CFO (Reestruturação & M&A)":
    jd_padrao = vaga_2_txt
else:
    jd_padrao = ""

# Colunas de Entrada e Saída
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Desafio da Posição")
    jd_input = st.text_area(
        "Descreva os requisitos e contexto da vaga:",
        value=jd_padrao,
        height=300,
        placeholder="Cole o perfil da vaga aqui..."
    )
    # Botão com ícone de lupa
    btn_mapear = st.button("🔍 Buscar Executivos")

with col_output:
    st.subheader("Resultado do Mapeamento")
    
    if btn_mapear:
        if not jd_input.strip():
            st.warning("Insira o descritivo da posição para realizar a busca.")
        else:
            with st.spinner("Consultando base de talentos e gerando parecer..."):
                try:
                    retriever = carregar_retriever()
                    docs_recuperados = retriever.invoke(jd_input)
                    
                    contexto_txt = "\n\n".join([
                        f"Nome: {doc.metadata.get('nome', 'Não informado')}\nPerfil Profissional: {doc.page_content}"
                        for doc in docs_recuperados
                    ])
                    
                    relatorio = elaborar_parecer_executivo(contexto_txt, jd_input)
                    
                    st.markdown(f'<div class="executive-card">{relatorio}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Erro ao processar mapeamento: {e}")
    else:
        st.info("Insira a vaga ao lado e clique em **🔍 Buscar Executivos** para visualizar os resultados.")