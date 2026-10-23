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
    model="gemini-3.5-flash-lite",
    temperature=0
)

# 4. Construção do Prompt e Orquestração (LCEL)
prompt_template = """
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