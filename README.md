# 🛢️ Commodity Intelligence Terminal

Plataforma institucional de pesquisa quantitativa e monitoramento de
commodities — Energia, Metais e Agricultura — construída em **Python +
Streamlit**, com arquitetura modular (Clean Architecture / SOLID),
fallback automático de dados e pronta para deploy imediato.

> **Status:** núcleo funcional em produção (v1.0). Módulos avançados
> (NLP, ESG, Supply Chain, geopolítica em mapa, deep learning) estão no
> roadmap — ver seção [Roadmap](#-roadmap-de-expansão) abaixo.

---

## ✅ O que está implementado (v1.0)

| Módulo | Conteúdo |
|---|---|
| 🌍 **Dashboard Global** | 22 ativos (Energia/Metais/Agri), preço, variação 1D/1S/1M/YTD, volatilidade, Sharpe, Sortino, Calmar, drawdown, momentum, tendência, treemap de performance |
| 🛢️⚙️🌾 **Analytics Setoriais** | Cards, tabela completa de métricas, candlestick individual, retorno acumulado, correlação intra-setor |
| 🇧🇷 **Commodities Brasileiras** | Petróleo, minério, soja, milho, café, açúcar com tabela de referência de mercados/portos |
| 🔗 **Macro & Correlações** | Heatmap commodities × DXY/Treasury/Fed Funds/CPI/PPI/Industrial Production, correlação rolante, PCA |
| ⚠️ **Risk Analytics** | VaR histórico e paramétrico, CVaR/Expected Shortfall, stress test customizável, histograma de retornos |
| 📈 **Forecast** | Monte Carlo (block bootstrap), cenários Base/Otimista/Pessimista, fan chart P10–P90, baseline de regressão (Linear/Ridge/Lasso) |
| 🧮 **Quant Research** | GARCH(1,1) próprio via MLE (scipy), volatilidade EWMA, walk-forward validation comparando modelos de tendência |
| ⚙️ **Infraestrutura** | Cache + retry automático, fallback sintético transparente, logging estruturado, testes unitários, Docker, configs para 4 plataformas de deploy |

Todos os gráficos são Plotly com tema escuro institucional, zoom, hover e exportação nativos.

---

## 🏗️ Arquitetura

```
commodity_terminal/
├── app.py                    # Entry point + tema escuro + landing page
├── pages/                    # Streamlit multipage (auto-descoberta)
│   ├── 1_🌍_Dashboard_Global.py
│   ├── 2_🛢️_Energy.py
│   ├── 3_⚙️_Metals.py
│   ├── 4_🌾_Agriculture.py
│   ├── 5_🇧🇷_Brazil.py
│   ├── 6_🔗_Macro_Correlations.py
│   ├── 7_⚠️_Risk_Analytics.py
│   ├── 8_📈_Forecast.py
│   └── 9_🧮_Quant_Research.py
├── config/settings.py         # Universo de ativos, séries macro, tema, parâmetros
├── data/
│   ├── data_manager.py        # Camada única de acesso a dados (cache + fallback)
│   └── sources/                # yahoo_finance.py · fred.py · synthetic.py
├── analytics/                  # metrics · correlation · risk · volatility (GARCH)
├── forecasting/models.py       # Regressão de tendência + Monte Carlo
├── charts/plotly_charts.py     # Biblioteca de gráficos com tema consistente
├── utils/                       # logger · sector_view (layout compartilhado)
├── tests/                       # pytest — metrics, risk, forecasting, correlation
├── requirements.txt / requirements-dev.txt
├── Dockerfile / docker-compose.yml
├── .streamlit/config.toml
└── .env.example
```

**Princípio central:** nenhuma página chama `yfinance` ou `requests`
diretamente — tudo passa por `data/data_manager.py`. Trocar de provedor
(ex: migrar para Bloomberg/Refinitiv no futuro) é uma mudança isolada
nesse arquivo, sem tocar em analytics ou páginas.

**Resiliência de dados:** toda fonte externa tem fallback sintético
automático (GBM com regimes de volatilidade, âncorado em preços-base
realistas). O usuário sempre vê um badge 🔸 quando os dados exibidos são
simulados — nunca um erro que quebra a página, e nunca dado simulado
apresentado silenciosamente como real.

---

## 🚀 Rodando localmente

```bash
git clone <seu-repositorio>
cd commodity_terminal

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
# edite .env e adicione sua FRED_API_KEY (gratuita) para dados macro reais

streamlit run app.py
```

Acesse `http://localhost:8501`. Sem `FRED_API_KEY` ou sem internet, o
terminal funciona normalmente com dados sintéticos (badge 🔸 visível).

### Rodando os testes

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## ☁️ Guia de Deploy

### 1. Streamlit Community Cloud (mais simples, grátis)
1. Suba o repositório no GitHub (público ou privado)
2. Acesse [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Selecione o repo, branch e `app.py` como arquivo principal
4. Em **Advanced settings → Secrets**, cole o conteúdo do seu `.env`:
   ```toml
   FRED_API_KEY = "sua_chave_aqui"
   ```
5. Deploy — pronto em ~2 minutos.

### 2. Render
1. **New → Web Service** → conecte o repositório
2. Runtime: `Docker` (usa o `Dockerfile` incluído automaticamente)
3. Configure as variáveis de ambiente do `.env.example` em **Environment**
4. Health Check Path: `/_stcore/health`
5. Deploy automático a cada push.

### 3. Railway
1. **New Project → Deploy from GitHub repo**
2. Railway detecta o `Dockerfile` automaticamente
3. Em **Variables**, adicione as chaves do `.env.example`
4. Em **Settings → Networking**, gere um domínio público na porta `8501`

### 4. Hugging Face Spaces
1. Crie um novo Space → SDK: **Docker**
2. Faça upload do repositório (ou conecte via Git)
3. O `Dockerfile` já expõe a porta `8501`, compatível com Spaces
4. Configure `FRED_API_KEY` em **Settings → Repository secrets**

### 5. Docker (self-hosted / qualquer VPS)
```bash
docker compose up -d --build
```
A aplicação sobe em `http://localhost:8501`. Logs: `docker compose logs -f`.

---

## 🗺️ Roadmap de Expansão

O escopo institucional completo (Bloomberg-grade) inclui dezenas de
módulos que dependem de fontes de dados pagas ou de infraestrutura mais
pesada (MLflow, deep learning). A base atual foi desenhada para que cada
item abaixo seja uma **extensão**, não uma refatoração:

**Dados fundamentais (requer chaves de API adicionais):**
- EIA Weekly Inventory, Rig Count, produção OPEP/EUA → `EIA_API_KEY`
- USDA NASS/QuickStats (safras, estoques, exportações) → `USDA_NASS_API_KEY`
- Curvas de futuros multi-vencimento (contango/backwardation reais) → CME DataMine / ICE
- PMI (ISM), Baltic Dry Index → Trading Economics / IHS Markit
- ComexStat/Secex + ANTAQ para fluxo real de exportação brasileira

**Modelos quantitativos adicionais** (`analytics/`, `forecasting/`):
- EGARCH, DCC-GARCH (pacote `arch`) — o GARCH(1,1) atual é a base extensível
- VAR / VECM / Cointegração (`statsmodels`)
- Kalman Filter, Hidden Markov Models
- XGBoost / LightGBM / CatBoost / Random Forest com SHAP values
- LSTM / GRU / Temporal Fusion Transformer (`torch`)
- Prophet para forecasting sazonal

**Novos módulos de página:**
- 🗺️ Geopolítica (mapas de rotas marítimas, estreitos, sanções — `plotly` + geodata)
- 📰 NLP de notícias (sentimento, entidades, timeline) — requer LLM/API de notícias
- 🌐 Supply Chain (Sankey de fluxo global, network graph)
- 🌱 ESG (emissões, intensidade de carbono)
- 🔁 MLflow — model registry, drift detection, retraining automático

**Infraestrutura:**
- Migração de cache em memória para Redis (multi-usuário/multi-processo)
- DuckDB/Polars para painéis de dados maiores
- CI/CD (GitHub Actions rodando `pytest` a cada push)

---

## 📄 Licença de Uso de Dados

Preços via Yahoo Finance (`yfinance`) e séries macro via FRED estão
sujeitos aos termos de uso de cada provedor — este projeto não redistribui
dados, apenas consulta as APIs em tempo real no ambiente do usuário.
