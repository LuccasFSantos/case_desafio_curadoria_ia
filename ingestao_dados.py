import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

candidatos = [
    {
        "id": "A", 
        "nome": "Ana Silva", 
        "texto": "15 anos de experiência no mercado financeiro. Ex-CFO de fintech unicórnio. Especialista em M&A, reestruturação financeira e captação de recursos (Series B e C). Perfil pragmático, voltado para resultados rápidos. Formação em Engenharia (Poli-USP) e MBA em Stanford."
    },
    {
        "id": "B", 
        "nome": "Bruno Costa", 
        "texto": "20 anos de experiência em Tl e Transformação Digital. Atual CIO em banco de varejo tradicional. Liderou a migração completa de sistemas legados para Cloud (GCP/AWS) gerindo equipes de +500 pessoas. Perfil apaziguador, forte cultura de governança e estabilidade. Ciência da Computação (UNICAMP)."
    },
    {
        "id": "C", 
        "nome": "Carolina Mendes", 
        "texto": "12 anos em tech. CTO em startup de logística. Foco absoluto em Inteligência Artificial, dados e otimização de rotas. Hands-on, programa em Python e escala times ágeis do zero. Perfil inovador, alta adaptabilidade, acostumada a ambientes de alta pressão e pouca estrutura. Matemática (UFRJ)."
    },
    {
        "id": "D", 
        "nome": "Diego Souza", 
        "texto": "18 anos em finanças corporativas na indústria pesada. CFO focado em controle de custos, compliance e auditoria. Liderou implantação de SAP globalmente. Perfil extremamente conservador, focado em governança rigorosa. Economia (Insper)."
    }
]

docs = [
    Document(page_content=c["texto"], metadata={"nome": c["nome"], "id": c["id"]}) 
    for c in candidatos
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Ingestão concluída './chroma_db'")