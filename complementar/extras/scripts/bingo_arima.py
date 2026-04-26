# ===============================================
# BINGO ARIMA — versão Python (statsforecast)
# ===============================================
#
# Objetivo: descobrir a ordem (p, d, q) do ARIMA por análise descritiva,
# antes de revelar o gabarito.
#
# Fluxo da brincadeira:
#   1. Rode "Sorteio + simulação" para gerar uma série nova.
#   2. Faça os passos 1-5 (descritiva, testes, diferenciação) e proponha
#      uma ordem (p, d, q).
#   3. Compare com o stepwise e o search do AutoARIMA.
#   4. Rode o bloco "Gabarito" no final pra revelar o que foi sorteado.
#
# Equivalente em R: complementar/extras/scripts/bingo_arima.R

# %% ===== 0. imports =====================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, ARIMA

import warnings
warnings.filterwarnings("ignore")

# Para um sorteio reprodutível, fixe a seed (ex.: np.random.default_rng(42))
RNG = np.random.default_rng()

# %% ===== 1. sorteio + simulação =========================================
# Mesma família do script R: ARIMA(p, d, q) com p,q ∈ {0,1,2} e d ∈ {0,1}
# e coeficientes uniformes em [0.3, 0.4].
p_true = int(RNG.integers(0, 3))
d_true = int(RNG.integers(0, 2))
q_true = int(RNG.integers(0, 3))

ar_parm = RNG.uniform(0.3, 0.4, size=p_true) if p_true else np.array([])
ma_parm = RNG.uniform(0.3, 0.4, size=q_true) if q_true else np.array([])

# ArmaProcess usa convenção (1 - phi_1 z - ...) Y_t = (1 + theta_1 z + ...) eps_t
# então passamos ar coefs com sinal negativo.
proc = ArmaProcess(np.r_[1.0, -ar_parm], np.r_[1.0, ma_parm])
y_stat = proc.generate_sample(nsample=300, scale=1.0)

# Aplicar d diferenças *integrais* (cumsum d vezes) para gerar I(d)
y = y_stat.copy()
for _ in range(d_true):
    y = np.cumsum(y)

dates = pd.date_range("1995-05-01", periods=len(y), freq="MS")
dados = pd.DataFrame({"unique_id": "vendas", "ds": dates, "y": y})

# %% ===== 2. helpers de visualização =====================================
def gg_tsdisplay(series, title="", lag_max=30):
    """Equivalente simples ao feasts::gg_tsdisplay com plot_type='partial':
    série no topo + ACF e PACF embaixo."""
    s = pd.Series(series).dropna()
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(s.index, s.values, lw=0.9, color="#E50505")
    ax0.set_title(title)
    ax0.grid(alpha=0.3)
    ax1 = fig.add_subplot(gs[1, 0])
    plot_acf(s, ax=ax1, lags=lag_max)
    ax1.set_title("ACF")
    ax2 = fig.add_subplot(gs[1, 1])
    plot_pacf(s, ax=ax2, lags=lag_max, method="ywm")
    ax2.set_title("PACF")
    plt.tight_layout()
    plt.show()


def kpss_p(x):
    """KPSS p-valor (H0: estacionária)."""
    return kpss(np.asarray(x).astype(float), regression="c", nlags="auto")[1]


def ndiffs_kpss(x, alpha=0.05, max_d=2):
    """Estilo feasts::unitroot_ndiffs: aplica KPSS sucessivamente."""
    cur = pd.Series(x).dropna().values.astype(float)
    for d in range(max_d + 1):
        try:
            p = kpss_p(cur)
        except Exception:
            return d
        if p > alpha:  # KPSS não rejeita estacionariedade
            return d
        cur = np.diff(cur)
    return max_d


# %% ===== PASSO 1: descritiva da série original ==========================
serie = pd.Series(dados["y"].values, index=dados["ds"], name="vendas")
gg_tsdisplay(serie, title="Série original")

adf_stat, adf_p, *_ = adfuller(serie)
print(f"ADF: stat = {adf_stat:.3f}   p-valor = {adf_p:.4f}")
print(f"     {'rejeita H0 (estacionária)' if adf_p < 0.05 else 'não rejeita H0 (raiz unitária)'}")

# %% ===== PASSO 2: KPSS + ndiffs =========================================
print(f"KPSS p-valor   = {kpss_p(serie.values):.4f}    (H0: estacionária)")
print(f"ndiffs (KPSS)  = {ndiffs_kpss(serie.values)}   (qtas diferenças regulares sugeridas)")
# nsdiffs (Canova-Hansen / OCSB) requer pmdarima e não é necessário aqui
# (não estamos gerando sazonalidade); o AutoARIMA decide internamente abaixo.

# %% ===== PASSO 3: 1ª e 2ª diferenças ====================================
dif1 = serie.diff()
gg_tsdisplay(dif1, title="1ª diferença", lag_max=30)

dif2 = dif1.diff()
gg_tsdisplay(dif2, title="2ª diferença", lag_max=30)

# Roteiro do quadro:
#   PASSO 1: descritiva
#   PASSO 2: avaliar necessidade de diferença (regular)
#   PASSO 3: descritiva da série diferenciada, se necessário
#   PASSO 4: olhar ACF/PACF da diferenciada para chutar (p, q)
#   PASSO 5: comparar com AutoARIMA stepwise/search (próximo bloco)


# %% ===== Modelagem com statsforecast ====================================
# manual_consenso: ARIMA(p_chute, d_chute, q_chute) — preencha à mão!
P_CHUTE, D_CHUTE, Q_CHUTE = 0, 0, 0

sf = StatsForecast(
    models=[
        ARIMA(order=(P_CHUTE, D_CHUTE, Q_CHUTE), include_mean=True, alias="manual_consenso"),
        AutoARIMA(stepwise=True,  alias="stepwise"),
        AutoARIMA(stepwise=False, alias="search"),
    ],
    freq="MS",
    n_jobs=1,
)
sf.fit(df=dados)

# AICc por modelo (model_['arma'] = (p, q, P, Q, m, d, D))
print("\n--- AICc por modelo (menor é melhor) ---")
fitted_models = {m.alias: m for m in sf.fitted_[0]}
for alias, m in fitted_models.items():
    info = m.model_
    aicc = info.get("aicc")
    arma = info.get("arma")
    ord_str = f"({arma[0]},{arma[5]},{arma[1]})" if arma is not None else "?"
    aicc_str = f"{aicc:.2f}" if isinstance(aicc, (int, float)) else "?"
    print(f"  {alias:18}  ordem = {ord_str:9}  AICc = {aicc_str}")

# Resíduos do stepwise — diretamente do model_
res_step = pd.Series(
    fitted_models["stepwise"].model_["residuals"],
    index=dados["ds"],
)
gg_tsdisplay(
    res_step,
    title="Resíduos · stepwise (deveriam parecer white noise)",
    lag_max=30,
)


# %% ===== GABARITO =======================================================
# (rode só depois que todo mundo arriscou um chute)
print("\n" + "=" * 44)
print(f"  Gabarito: ARIMA({p_true}, {d_true}, {q_true})")
print(f"  ar_parm = {np.round(ar_parm, 3).tolist()}")
print(f"  ma_parm = {np.round(ma_parm, 3).tolist()}")
print("=" * 44)
