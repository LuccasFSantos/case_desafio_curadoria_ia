
# Executive Talent Matchmaking System (RAG + Gemini)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![GCP](https://img.shields.io/badge/Cloud-GCP%20Vertex%20AI-4285F4.svg)](https://cloud.google.com/)
[![LLM Engine](https://img.shields.io/badge/Model-Gemini%201.5%20Pro-8E44AD.svg)](https://deepmind.google/technologies/gemini/)
[![Framework](https://img.shields.io/badge/Orchestration-LangChain-00A67E.svg)](https://www.langchain.com/)

Sistema Inteligente End-to-End projetado para a curadoria preditiva de talentos C-Level e Alta Gestão em consultorias de Executive Search. A aplicação realiza busca semântica em acervos não estruturados de currículos executivos e gera justificativas consultivas fundamentadas para suporte à tomada de decisão dos sócios.

---

## 🏛️ Desenho Arquitetural em Nuvem (GCP - Produção)

```text
                                  [ PIPELINE DE PRODUÇÃO NO GCP ]

   +-------------------+      +--------------------+      +-----------------------+
   |  Currículos PDFs  | ---> |  GCS (Cloud Storage| ---> | Document AI (Parsing) |
   +-------------------+      +--------------------+      +-----------------------+
                                                                      |
                                                                      v
                                                          +-----------------------+
                                                          | Cloud DLP (Anonimiza) |
                                                          +-----------------------+
                                                                      |
                                                                      v
   +-------------------+      +--------------------+      +-----------------------+
   | Job Description   | ---> | Cloud Run (App UI) | ---> | Vertex AI Embeddings  |
   +-------------------+      +--------------------+      +-----------------------+
                                        |                             |
                                        v                             v
                              +--------------------+      +-----------------------+
                              | Gemini 1.5 Pro     | <--- | Vertex AI Vector Search|
                              | (LLM Engine)       |      | (Base Vetorial k-NN)  |
                              +--------------------+      +-----------------------+

```

### Componentes da Arquitetura:

1. **Cloud Storage (GCS):** Ingestão e repositório seguro para documentos de currículos brutos (PDF/Word).
2. **Document AI:** Extração inteligente de texto não estruturado, preservando layouts, seções e tabelas.
3. **Cloud Data Loss Prevention (DLP):** Detecção e mascaramento automático de dados sensíveis e PII (Nomes, telefones, e-mails).
4. **Vertex AI Embeddings (`text-embedding-004`):** Conversão semântica de seções e competências executivas em vetores de alta dimensão.
5. **Vertex AI Vector Search:** Banco vetorial Serverless gerenciado, de baixíssima latência (sub-10ms) para busca de similaridade semântica.
6. **Gemini 1.5 Pro via Vertex AI:** LLM corporativo responsável pela geração da análise consultiva estratégica sob parâmetros rígidos e `temperature=0.0`.
7. **Cloud Run:** Hospedagem containerizada e auto-escalável da interface web Streamlit para os sócios.

---

## Stack Tecnológica

* **Linguagem Principal:** Python 3.11+
* **LLM Engine:** Google Gemini 1.5 Pro (`google-genai` SDK)
* **Vector Database:** ChromaDB (Persistência Vetorial Local) / Vertex AI Vector Search (Cloud)
* **Modelos de Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
* **Orquestração RAG:** LangChain Expression Language (LCEL)
* **Frontend / Interface:** Streamlit (UI Executiva de Alta Maturidade Visual)

---

## Segurança, Governança e Anti-Alucinação

1. **Privacidade e Proteção de PII:**
* A arquitetura foi concebida sob o princípio de *Privacy by Design*.
* Na versão Cloud, os PIIs são tokenizados antes da vetorização via **Cloud DLP**.
* O uso da API Enterprise da Vertex AI garante formalmente a **não retenção de dados** e impede o uso de informações corporativas para treinamento público de modelos.


2. **Mitigação de Alucinações:**
* **Parâmetros Determinísticos:** Uso de `temperature=0.0` no modelo Gemini.
* **Grounding Rígido:** Instruções de sistema forçam o modelo a emitir pareceres baseados **exclusivamente** nos trechos recuperados pelo RAG.
* **Detecção de Falta de Dados:** Caso uma competência do JD não esteja no CV, o agente é instruído a explicitar a ausência.



---

##  Guia de Instalação e Execução Local

### 1. Pré-requisitos

* Python 3.11 instalado.
* Chave de API do Google Gemini (`GEMINI_API_KEY`).

### 2. Clonar o Repositório e Criar Ambiente Virtual

```bash
git clone [https://github.com/LuccasFSantos/desafio_curadoria_ia.git](https://github.com/LuccasFSantos/desafio_curadoria_ia.git)
cd desafio_curadoria_ia

python -m venv venv

```

**Ativação do Virtualenv:**

* **Windows:** `venv\Scripts\activate`
* **Linux/Mac:** `source venv/bin/activate`

### 3. Instalar Dependências

```bash
pip install -r requirements.txt

```

### 4. Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
GEMINI_API_KEY="COLE_SUA_CHAVE_API_AQUI"

```

### 5. Ingestão da Base Vetorial de Candidatos

Execute o pipeline para povoar o banco vetorial local (ChromaDB) com os mini-currículos:

```bash
python ingestao_dados.py

```

### 6. Inicializar a Aplicação Web

Inicie o servidor local do Streamlit:

```bash
streamlit run app.py

```

---

##  Validação dos Casos de Teste (Matchmaking)

A aplicação inclui botões de testes rápidos na sidebar para os dois cenários do desafio:

1. **Vaga 1 (Tech Lead / CTO - IA & Dados):**
* **Resultado Esperado:** Indicação prioritária da **Candidata C (Carolina Mendes)**.
* **Fundamentação:** Experiência prévia como CTO em startup, perfil *hands-on*, código em Python, forte histórico em IA/Dados e adaptabilidade a ambientes dinâmicos.


2. **Vaga 2 (CFO - Reestruturação & Captação):**
* **Resultado Esperado:** Indicação prioritária da **Candidata A (Ana Silva)**.
* **Fundamentação:** Ex-CFO de fintech unicórnio, histórico comprovado em M&A, reestruturação financeira e rodadas avançadas de captação (Series B e C).



---

## Limitações do Protótipo e Próximos Passos

| Limitação Atual | Próximo Passo para Produção |
| --- | --- |
| Banco vetorial ChromaDB local | Migração para **Vertex AI Vector Search** gerenciado |
| Embeddings locais com HuggingFace | Substituição pelo modelo nativo **Vertex AI Embeddings** |
| Triagem estática via formulário | Suporte a upload dinâmico de múltiplos arquivos PDF de currículos |
| Avaliação qualitativa manual | Implementação de avaliação contínua via **RAGAS** e **LLM-as-a-Judge** |
