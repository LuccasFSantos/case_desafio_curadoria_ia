import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Carregamento de Variáveis
load_dotenv()

# 2. Configuração do Retriever Local
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Configuração do LLM (LangChain Wrapper)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)

# 4. Construção do Prompt e Orquestração (LCEL)
prompt_template = """
Você é um Sênior Partner de uma consultoria de recrutamento executivo de alto nível.
O seu objetivo é apoiar a tomada de decisão avaliando candidatos para uma vaga C-Level.

Com base EXCLUSIVAMENTE nos perfis dos candidatos fornecidos abaixo, avalie o fit para a vaga descrita.
Para cada um dos 3 candidatos recuperados, escreva um parágrafo analítico, estratégico e consultivo justificando o porquê do perfil (Hard Skills e Soft Skills) fazer sentido para o desafio. 
Seja direto, profissional e não invente habilidades que não constem no perfil.

Perfis Recuperados da Base:
{context}

Descrição da Vaga (Job Description):
{question}

Análise Consultiva:
"""
prompt = PromptTemplate.from_template(prompt_template)

def formatar_documentos(docs):
    return "\n\n".join(f"Candidato: {doc.metadata['nome']}\nPerfil: {doc.page_content}" for doc in docs)

rag_chain = (
    {"context": retriever | formatar_documentos, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Teste de Validação
vaga_1 = """
Vaga 1 (Tech Lead / CTO): Startup B2B de rápido crescimento busca CTO com forte viés em IA e dados. 
Necessário perfil hands-on para escalar o time técnico e construir arquitetura escalável do zero. 
Ambiente dinâmico, exige alta adaptabilidade e inovação.
"""

print("Analisando candidatos para a Vaga 1...\n")
resposta_analise = rag_chain.invoke(vaga_1)
print(resposta_analise)