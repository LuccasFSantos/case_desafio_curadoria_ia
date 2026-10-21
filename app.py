import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuração da página e variáveis de ambiente
load_dotenv()
st.set_page_config(
    page_title="Executive Talent Matching | Intelligence System",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada para um design executivo e moderno
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .candidate-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #0F172A;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1E293B;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Cache de carregamento do Banco Vetorial
@st.cache_resource
def carregar_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3})

# Função para executar a geração via SDK oficial google-genai
def gerar_analise_curadoria(contexto: str, job_description: str) -> str:
    client = genai.Client()
    
    prompt_sistema = """
Você é um Sênior Partner de uma consultoria de recrutamento executivo de alto nível (partnership horizontal).
Seu objetivo é gerar uma análise consultiva, estratégica e altamente analítica avaliando os candidatos para a vaga informada.

INSTRUÇÕES E DIRETRIZES:
1. Avalie rigorosamente os candidatos recuperados da base contra o Job Description.
2. Para cada um dos 3 candidatos, apresente:
   - Nome e Perfil Atual
   - Análise Estratégica (Hard Skills & Soft Skills aplicadas ao desafio da vaga)
   - Veredicto de Fit (Alto, Médio ou Baixo)
3. Não invente ou extrapole habilidades que não constem estritamente nos perfis fornecidos.
4. Utilize tom estritamente executivo e consultivo.
"""
    
    conteudo_prompt = f"""
{prompt_sistema}

PERFIS RECUPERADOS DA BASE DE TALENTOS:
{contexto}

JOB DESCRIPTION (DESCRITIVO DA VAGA):
{job_description}

ANÁLISE CONSULTIVA EXECUTIVA:
"""
    
    config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=0.95,
    )
    
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=conteudo_prompt,
        config=config
    )
    return response.text

# Interface do Usuário (Streamlit)
st.markdown("<h1 class='main-header'>👔 Executive Talent Matchmaking</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Curadoria preditiva de altos executivos baseada em RAG semântico e LLM</p>", unsafe_allow_html=True)

# Sidebar para seleção de casos de teste
st.sidebar.header("📋 Casos de Teste Rápidos")
opcao_vaga = st.sidebar.selectbox(
    "Escolha uma vaga pré-configurada:",
    ["Personalizado", "Vaga 1: Tech Lead / CTO (IA & Dados)", "Vaga 2: CFO (Reestruturação & M&A)"]
)

vaga_1_txt = "Startup B2B de rápido crescimento busca CTO com forte viés em IA e dados. Necessário perfil hands-on para escalar o time técnico e construir arquitetura escalável do zero. Ambiente dinâmico, exige alta adaptabilidade e inovação."
vaga_2_txt = "Empresa de tecnologia em fase de alto crescimento busca CFO para liderar próxima rodada de captação institucional e preparar a empresa para futuro M&A. Exige experiência prévia com startups em estágios avançados e relacionamento com fundos de VC."

if opcao_vaga == "Vaga 1: Tech Lead / CTO (IA & Dados)":
    jd_default = vaga_1_txt
elif opcao_vaga == "Vaga 2: CFO (Reestruturação & M&A)":
    jd_default = vaga_2_txt
else:
    jd_default = ""

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📄 Job Description (JD)")
    jd_input = st.text_area(
        "Insira a descrição completa da posição executiva:",
        value=jd_default,
        height=280,
        placeholder="Cole aqui o perfil desejado da vaga..."
    )
    btn_executar = st.button("🔍 Executar Matchmaking Executivo")

with col2:
    st.subheader("📊 Relatório de Curadoria Executiva")
    if btn_executar:
        if not jd_input.strip():
            st.warning("Por favor, insira uma descrição de vaga válida.")
        else:
            with st.spinner("Realizando busca semântica no vector store e gerando análise consultiva..."):
                try:
                    retriever = carregar_vector_store()
                    docs_recuperados = retriever.invoke(jd_input)
                    
                    contexto_txt = "\n\n".join([
                        f"Candidato: {doc.metadata.get('nome', 'N/A')}\nPerfil: {doc.page_content}"
                        for doc in docs_recuperados
                    ])
                    
                    resultado = gerar_analise_curadoria(contexto_txt, jd_input)
                    
                    st.markdown(resultado)
                    
                except Exception as e:
                    st.error(f"Erro durante o processamento da curadoria: {e}")
    else:
        st.info("Insira a vaga e clique no botão para visualizar o Top 3 candidatos e parecer consultivo.") 