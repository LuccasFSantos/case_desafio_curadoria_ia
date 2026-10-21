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

### 2. Ativação do Virtualenv

- **Windows:**
  ```cmd
  venv\Scripts\activate
  ```
- **Linux/Mac:**
  ```bash
  source venv/bin/activate
  ```

---

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

---

### 4. Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
GEMINI_API_KEY="COLE_SUA_CHAVE_API_AQUI"
```

---

### 5. Ingestão da Base Vetorial de Candidatos

Execute o pipeline para povoar o banco vetorial local (ChromaDB) com os mini-currículos:

```bash
python ingestao_dados.py
```

---

### 6. Inicializar a Aplicação Web

Inicie o servidor local do Streamlit:

```bash
streamlit run app.py
```

---

## 🧪 Validação dos Casos de Teste (Matchmaking)

A aplicação inclui botões de testes rápidos na *sidebar* para os dois cenários do desafio:

1. **Vaga 1 (Tech Lead / CTO - IA & Dados):**
   - **Resultado Esperado:** Indicação prioritária da **Candidata C (Carolina Mendes)**.
   - **Fundamentação:** Experiência prévia como CTO em startup, perfil *hands-on*, conhecimento em Python, forte histórico em IA/Dados e alta adaptabilidade a ambientes dinâmicos.

2. **Vaga 2 (CFO - Reestruturação & Captação):**
   - **Resultado Esperado:** Indicação prioritária da **Candidata A (Ana Silva)**.
   - **Fundamentação:** Ex-CFO de fintech unicórnio, histórico comprovado em M&A, reestruturação financeira e rodadas avançadas de captação (Series B e C).