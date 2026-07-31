import streamlit as st
import pandas as pd
import numpy as np

from config.settings import ALL_ASSETS, APP_NAME
from data.data_manager import load_price_history
from analytics import risk, metrics, backtesting, regimes
from charts import plotly_charts as charts

# NOTA v4.5.0: st.set_page_config() removido daqui — agora é chamado
# uma única vez em app.py, antes de st.navigation(...).run().

st.title("⚠️ Risk Analytics")
st.caption(
    "VaR histórico e paramétrico, backtesting formal (Kupiec/Christoffersen), detecção de "
    "regimes de volatilidade via Hidden Markov Model e stress test."
)

# --------------------------------------------------------------------------
# CONTROLES COMPARTILHADOS (aplicam-se a todas as abas)
# --------------------------------------------------------------------------
asset_names = {a.name: a for a in ALL_ASSETS}
selected_name = st.selectbox("Ativo", list(asset_names.keys()))
asset = asset_names[selected_name]

col_conf, col_window = st.columns(2)
with col_conf:
    confidence = st.select_slider("Nível de confiança", options=[0.90, 0.95, 0.975, 0.99], value=0.95)
with col_window:
    window = st.slider("Janela histórica (pregões)", 60, 500, 252, step=20)

with st.spinner("Carregando dados..."):
    pdat = load_price_history(asset)

if pdat.is_synthetic:
    st.warning("⚠️ Exibindo **dados simulados** — fonte ao vivo indisponível neste ambiente.", icon="⚠️")

close = pdat.df["Close"]

st.divider()

tab_overview, tab_backtest, tab_regimes = st.tabs([
    "📊 Visão Geral (VaR/CVaR/Stress)",
    "🎯 Backtesting de VaR",
    "🔄 Regimes de Volatilidade (HMM)",
])

# ============================================================================
# ABA 1 — VISÃO GERAL
# ============================================================================
with tab_overview:
    r = risk.risk_summary(close, confidence=confidence, window=window)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"VaR Histórico ({confidence:.1%})", f"{r['var_historico']:.2%}")
    c2.metric(f"VaR Paramétrico ({confidence:.1%})", f"{r['var_parametrico']:.2%}")
    c3.metric("CVaR / Expected Shortfall", f"{r['cvar']:.2%}")

    st.caption(
        f"Interpretação: com {confidence:.0%} de confiança, a perda diária não deve exceder "
        f"**{r['var_historico']:.2%}** do valor posicionado (janela de {window} pregões). O CVaR "
        f"informa a perda média condicional nos cenários que ultrapassam esse limite."
    )

    st.divider()

    st.subheader("Risco Ajustado ao Retorno")
    row = metrics.summary_row(close, window=window)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Volatilidade Anualizada", f"{metrics.annualized_volatility(close, window=window):.2%}")
    m2.metric("Sharpe Ratio (252d)", f"{metrics.sharpe_ratio(close, window=252):.2f}")
    m3.metric("Sortino Ratio (252d)", f"{metrics.sortino_ratio(close, window=252):.2f}")
    m4.metric("Máximo Drawdown (252d)", f"{metrics.max_drawdown(close.tail(252)):.2%}")

    st.divider()

    st.subheader("Stress Test — Choques Instantâneos")
    custom_shocks = st.text_input("Choques (%), separados por vírgula", "-30,-20,-10,-5,5,10,20")
    try:
        shocks = [float(x.strip()) / 100 for x in custom_shocks.split(",") if x.strip()]
    except ValueError:
        shocks = None
        st.error("Formato inválido — use números separados por vírgula, ex: -30,-10,10,20")

    if shocks:
        stress_df = risk.stress_test(close, shocks_pct=shocks)
        st.dataframe(stress_df, use_container_width=True, hide_index=True)
        st.plotly_chart(
            charts.bar_chart(stress_df["choque"].tolist(), stress_df["variacao_absoluta"].tolist(),
                              title="Impacto Absoluto por Cenário de Choque"),
            use_container_width=True,
        )

    st.divider()

    st.subheader("Distribuição de Retornos Diários")
    rets = metrics.daily_returns(close).tail(window)
    st.plotly_chart(
        charts.histogram_chart(rets.values, title="Histograma de Retornos", x_title="Retorno diário"),
        use_container_width=True,
    )

# ============================================================================
# ABA 2 — BACKTESTING DE VaR (Kupiec / Christoffersen)
# ============================================================================
with tab_backtest:
    st.markdown(
        "Testa formalmente se o modelo de VaR está **calibrado**: a taxa de exceções observada "
        "bate com a esperada (Kupiec) e as exceções não se concentram em clusters (Christoffersen). "
        "Metodologia padrão de validação de modelos em risk management institucional (Basel)."
    )

    method = st.radio("Método de VaR a testar", ["historical", "parametric"],
                       format_func=lambda x: "Histórico" if x == "historical" else "Paramétrico (Normal)",
                       horizontal=True)

    n_obs_available = len(metrics.daily_returns(close))
    if n_obs_available < window + 60:
        st.warning(
            f"⚠️ Histórico insuficiente para um backtest robusto: {n_obs_available} observações "
            f"disponíveis, mínimo recomendado ≈ {window + 60} (janela de estimação + amostra de teste)."
        )
    else:
        with st.spinner("Rodando VaR rolling out-of-sample e testes de backtest..."):
            bt = backtesting.joint_backtest(close, confidence=confidence, window=window, method=method)

        kupiec, christoffersen, joint = bt["kupiec"], bt["christoffersen"], bt["joint"]

        st.subheader("Resultado dos Testes")
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Exceções Observadas", f"{kupiec['n_breaches']} / {kupiec['n_obs']}",
                   f"taxa obs. {kupiec['breach_rate']:.2%} vs. esperada {kupiec['expected_rate']:.2%}")
        bc2.metric("Kupiec (POF) p-valor", f"{kupiec['p_value']:.3f}" if pd.notna(kupiec['p_value']) else "N/D",
                   "✅ modelo calibrado" if kupiec['reject_h0'] is False else ("❌ rejeita H0" if kupiec['reject_h0'] else None))
        bc3.metric("Christoffersen (Indep.) p-valor",
                   f"{christoffersen['p_value']:.3f}" if pd.notna(christoffersen['p_value']) else "N/D",
                   "✅ exceções independentes" if christoffersen['reject_h0'] is False else ("❌ há clustering" if christoffersen['reject_h0'] else None))

        verdict = "✅ Modelo aprovado nos dois testes (nível 5%)"
        if joint["reject_h0"] is True:
            verdict = "❌ Modelo rejeitado no teste conjunto (nível 5%) — VaR mal calibrado para este ativo/janela"
        elif joint["reject_h0"] is None:
            verdict = "⚠️ Dados insuficientes para conclusão estatística"
        st.info(
            f"**Teste conjunto (Christoffersen 1998):** LR = {joint['lr_stat']:.2f}, "
            f"p-valor = {joint['p_value']:.3f} → {verdict}",
            icon="🎯",
        )

        st.divider()

        st.subheader("Retornos vs. VaR Previsto — Exceções Marcadas")
        st.plotly_chart(
            charts.var_breach_chart(metrics.daily_returns(close), bt["var_series"], bt["breaches"],
                                     title=f"{asset.name} — Backtest de VaR ({method}, {confidence:.0%})"),
            use_container_width=True,
        )
        st.caption(
            "Cada VaR do dia t é estimado usando **apenas** os `window` retornos anteriores a t "
            "(nunca olha o próprio dia) — evita viés otimista (look-ahead bias) no backtest."
        )

        with st.expander("📘 Como interpretar estes testes"):
            st.markdown(r"""
            **Kupiec POF (Proportion of Failures), Kupiec (1995):**
            Testa se a proporção de exceções observada é estatisticamente compatível com a
            taxa esperada \((1-\text{confiança})\). Estatística:
            $$
            LR_{POF} = -2 \ln\left[\frac{(1-p)^{n-x} p^x}{(1-\hat{\pi})^{n-x} \hat{\pi}^x}\right] \sim \chi^2(1)
            $$
            onde \(p\) é a taxa esperada, \(\hat{\pi} = x/n\) é a taxa observada, \(x\) o número de
            exceções e \(n\) o total de observações. **p-valor < 0.05 → rejeita o modelo** (excesso
            ou falta de exceções).

            **Christoffersen Independência (1998):**
            Um modelo pode ter a proporção certa de exceções, mas concentradas em clusters
            (ex: todas numa crise) — isso ainda é ruim, pois indica reação lenta a mudanças de
            volatilidade. O teste verifica se a probabilidade de exceção no dia \(t\) depende do
            que aconteceu em \(t-1\) (processo de Markov de 1ª ordem). **p-valor < 0.05 → há
            dependência temporal** (clustering).

            **Teste Conjunto (Cobertura Condicional):** \(LR_{CC} = LR_{POF} + LR_{IND} \sim \chi^2(2)\)
            — o modelo só passa se cobrir corretamente a magnitude **e** a independência.
            """)

# ============================================================================
# ABA 3 — REGIMES DE VOLATILIDADE (HMM)
# ============================================================================
with tab_regimes:
    st.markdown(
        "Identifica **regimes de mercado** (baixa vs. alta volatilidade) via Hidden Markov Model "
        "Gaussiano de 2 estados, ajustado por máxima verossimilhança (algoritmo Baum-Welch). "
        "Útil para entender em que tipo de mercado o ativo está agora e por que o VaR falha mais "
        "em certos períodos (aba anterior)."
    )

    n_obs = len(metrics.daily_returns(close))
    if n_obs < 100:
        st.warning(f"⚠️ Histórico insuficiente para ajustar o HMM: {n_obs} observações (mínimo ~100).")
    else:
        with st.spinner("Ajustando HMM de 2 estados via Baum-Welch..."):
            try:
                reg = regimes.regime_summary(close, n_iter=150)
                hmm_error = None
            except Exception as exc:  # noqa: BLE001
                reg = None
                hmm_error = str(exc)

        if hmm_error:
            st.error(f"Não foi possível ajustar o modelo: {hmm_error}")
        else:
            rc1, rc2 = st.columns(2)
            rc1.metric("Regime Atual", reg["current_regime_label"],
                       f"confiança {reg['current_regime_prob']:.1%}")
            persistence = np.diag(reg["transition_matrix"])
            avg_duration_days = 1 / (1 - persistence)
            rc2.metric("Duração Média do Regime Atual",
                       f"{avg_duration_days[1 if reg['current_regime_label'] == 'Alta Volatilidade' else 0]:.0f} pregões")

            st.divider()

            st.subheader("Preço com Regimes Sombreados")
            st.caption("Áreas em vermelho = regime de Alta Volatilidade (classificação Viterbi, mais provável).")
            st.plotly_chart(
                charts.regime_price_chart(close.tail(min(len(close), 1000)),
                                           reg["viterbi_states"].tail(min(len(close), 1000)),
                                           title=f"{asset.name} — Regimes de Volatilidade"),
                use_container_width=True,
            )

            st.plotly_chart(
                charts.regime_probability_chart(reg["state_probs"].tail(min(len(close), 1000))),
                use_container_width=True,
            )

            st.divider()

            st.subheader("Estatísticas por Regime")
            fmt_regime = {
                "Retorno médio diário": "{:.4%}",
                "Volatilidade anualizada": "{:.2%}",
                "Persistência (prob. de permanecer)": "{:.2%}",
            }
            st.dataframe(reg["regime_stats"].style.format(fmt_regime), use_container_width=True)

            st.subheader("Matriz de Transição")
            trans_df = pd.DataFrame(
                reg["transition_matrix"],
                index=["De: Baixa Vol", "De: Alta Vol"],
                columns=["Para: Baixa Vol", "Para: Alta Vol"],
            )
            st.dataframe(trans_df.style.format("{:.2%}"), use_container_width=True)

            with st.expander("📘 Como interpretar o HMM de regimes"):
                st.markdown(r"""
                **O que o modelo faz:** assume que os retornos diários são gerados por 2 processos
                gaussianos distintos (regimes), cada um com sua própria média e volatilidade, e que
                o mercado transita entre eles de forma probabilística (cadeia de Markov oculta —
                "oculta" porque não observamos o regime diretamente, só os retornos).

                **Ajuste:** algoritmo Baum-Welch (Expectation-Maximization) estima os parâmetros
                (médias, volatilidades, matriz de transição) que maximizam a verossimilhança dos
                dados observados. A sequência de regimes mostrada usa o **algoritmo de Viterbi**
                (a sequência única mais provável dada toda a série).

                **Persistência:** a diagonal da matriz de transição indica a probabilidade de o
                regime se manter no dia seguinte — valores próximos de 1 (ex: 0.97) implicam
                regimes duradouros (dezenas de pregões), típico de volatilidade em commodities.

                **Uso prático:** um VaR histórico calculado sobre uma janela que mistura os dois
                regimes tende a subestimar o risco em períodos de alta volatilidade recém-iniciados
                — é exatamente esse tipo de falha que o backtesting da aba anterior expõe.

                Referência: Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of
                Nonstationary Time Series and the Business Cycle.* Econometrica, 57(2), 357-384.
                """)