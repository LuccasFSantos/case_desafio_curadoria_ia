# 👔 Executive Talent Matchmaking System (RAG + Gemini)

Sistema Inteligente End-to-End desenvolvido para curadoria preditiva de talentos C-Level e Alta Gestão. A solução realiza busca semântica em base de currículos não estruturados e gera justificativas consultivas fundamentadas para sócios de executive search.

---

## 📐 Arquitetura da Solução em Produção (GCP Reference Architecture)

1. **Ingestão e Parsing:** Document AI processa os currículos bruto no Cloud Storage.
2. **Vetorização e Indexação:** Embeddings de alta dimensão gerenciados no **Vertex AI Vector Search**.
3. **Orquestração e Agente:** A interface no **Cloud Run** consulta a base vetorial e envia o contexto restrito ao **Gemini 1.5 Pro** via SDK `google-genai`.

---

## 🛠️ Stack Tecnológica

- **LLM Engine:** Google Gemini 1.5 Pro (`google-genai` SDK)
- **Vector Database:** ChromaDB (Persistência local)
- **Embeddings Model:** HuggingFace `all-MiniLM-L6-v2`
- **Frontend / Interface:** Streamlit (UI Executiva)
- **Orquestração:** Python / RAG Nativo

---

## 🛡️ Diretrizes de Segurança, Privacidade e Anti-Alucinação

1. **Proteção de PII:** Em ambiente produtivo, aplica-se a API do **Cloud DLP (Data Loss Prevention)** no pipeline de ingestão para anonimização prévia de nomes, e-mails e contatos de executivos.
2. **Mitigação de Alucinações:** O modelo opera com `temperature=0.0` e instruções de sistema estritas que impedem a dedução ou invenção de qualificações não listadas nos currículos.
3. **Isolamento de Dados:** Os dados não são utilizados para treinamento de modelos públicos e permanecem dentro da VPC do projeto.

---

## 🚀 Instruções para Execução Local

### 1. Clonar o Repositório e Configurar o Ambiente

```bash
git clone [https://github.com/LuccasFSantos/desafio_curadoria_ia.git](https://github.com/LuccasFSantos/desafio_curadoria_ia.git)
cd desafio_curadoria_ia
python -m venv venv