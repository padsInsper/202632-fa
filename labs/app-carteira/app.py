"""Monte sua Carteira: app do Lab 3 (Shiny for Python, compatível com shinylive).

Roda tanto local (`shiny run labs/app-carteira/app.py`) quanto embutido no site via
shinylive (Pyodide, no navegador). Para funcionar no navegador não usa `yfinance`: lê os
retornos pré-baixados de `dados_carteira.csv` (gerado por `fetch_dados.py`).

Fluxo da dinâmica:
1. Cada aluno responde às loterias (estilo Kahneman/Holt-Laury) e o app estima o seu
   coeficiente de aversão a risco (lambda) da utilidade média-variância U = mu - lambda/2 * sigma^2.
2. Os alunos escolhem ativos do catálogo (buscando por nome), mexem nos pesos e veem ao
   vivo retorno, volatilidade, Sharpe, VaR, a fronteira eficiente, o retorno acumulado vs.
   o Ibovespa e a matriz de correlação.
3. A fronteira marca a "carteira ótima do seu perfil": a que maximiza a utilidade dado o
   lambda elicitado. Lambda alto puxa para a mínima variância; lambda baixo, para mais retorno.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

BENCH = "^BVSP"

INSPER_RED = "#E50505"
INSPER_TURQUESA = "#3ACC9F"
INSPER_AMARELO = "#FFCC00"
INSPER_ROXO = "#730D9F"
INSPER_GRAY = "#5B5B5B"

# Loterias estilo Holt-Laury: em cada decisão, a "chance alta" p cresce. A opção A é
# sempre a mais segura (payoffs próximos) e a B a mais arriscada (payoffs distantes).
# Quanto mais vezes a pessoa fica no seguro (A), maior a aversão a risco.
LOTERIAS = [0.10, 0.30, 0.50, 0.70, 0.90, 1.00]

# Mapa didático do nº de escolhas seguras (0 a 6) para o lambda da utilidade
# média-variância. Não é uma estimativa precisa de CRRA: é uma calibração para que o
# perfil mova de forma visível o ponto ótimo ao longo da fronteira.
LAMBDA_MAP = {0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0, 4: 8.0, 5: 12.0, 6: 20.0}


def perfil_label(lam: float) -> str:
    if lam <= 2:
        return "tolerante ao risco"
    if lam <= 5:
        return "moderado"
    if lam <= 12:
        return "avesso ao risco"
    return "muito avesso ao risco"


def _load_returns() -> pd.DataFrame:
    """Lê o CSV empacotado. No shinylive ele é montado em 'dados_carteira.csv'."""
    candidatos = ["dados_carteira.csv"]
    try:
        candidatos.append(str(Path(__file__).with_name("dados_carteira.csv")))
    except NameError:
        pass
    candidatos.append("labs/app-carteira/dados_carteira.csv")
    for c in candidatos:
        p = Path(c)
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True)
    raise FileNotFoundError("dados_carteira.csv não encontrado em: " + ", ".join(candidatos))


RETORNOS = _load_returns()
ATIVOS_DISP = [c for c in RETORNOS.columns if c != BENCH]

# Nome das empresas/fundos por setor (só rótulos; o universo é o que está no CSV)
CATALOGO = {
    "Fundos imobiliários (FIIs)": {
        "HGRE11.SA": "CSHG Real Estate", "BTLG11.SA": "BTG Pactual Logística",
        "HGRU11.SA": "CSHG Renda Urbana", "VGIR11.SA": "Valora CRI",
        "KNRI11.SA": "Kinea Renda Imobiliária", "KNCR11.SA": "Kinea Rendimentos",
        "MXRF11.SA": "Maxi Renda", "HGLG11.SA": "CSHG Logística",
        "XPML11.SA": "XP Malls", "VISC11.SA": "Vinci Shopping Centers",
        "HGCR11.SA": "CSHG Recebíveis", "RECT11.SA": "REC Renda Imobiliária",
    },
    "Bancos e financeiro": {
        "ITUB4.SA": "Itaú Unibanco", "BBDC4.SA": "Bradesco", "BBAS3.SA": "Banco do Brasil",
        "SANB11.SA": "Santander Brasil", "ITSA4.SA": "Itaúsa", "B3SA3.SA": "B3 (bolsa)",
        "BPAC11.SA": "BTG Pactual",
    },
    "Commodities e materiais": {
        "PETR4.SA": "Petrobras (PN)", "PETR3.SA": "Petrobras (ON)", "VALE3.SA": "Vale",
        "PRIO3.SA": "PRIO (PetroRio)", "CSNA3.SA": "CSN", "GGBR4.SA": "Gerdau",
        "SUZB3.SA": "Suzano", "KLBN11.SA": "Klabin", "BRAP4.SA": "Bradespar",
        "CMIN3.SA": "CSN Mineração",
    },
    "Consumo e varejo": {
        "ABEV3.SA": "Ambev", "MGLU3.SA": "Magazine Luiza", "LREN3.SA": "Lojas Renner",
        "ASAI3.SA": "Assaí", "PCAR3.SA": "Pão de Açúcar", "NTCO3.SA": "Natura",
        "RADL3.SA": "Raia Drogasil", "CRFB3.SA": "Carrefour Brasil",
    },
    "Indústria e serviços": {
        "WEGE3.SA": "WEG", "EMBR3.SA": "Embraer", "RENT3.SA": "Localiza",
        "RAIL3.SA": "Rumo", "TOTS3.SA": "Totvs", "VIVT3.SA": "Vivo (Telefônica)",
    },
    "Energia e utilities": {
        "ELET3.SA": "Eletrobras", "EGIE3.SA": "Engie Brasil", "CPLE6.SA": "Copel",
        "TAEE11.SA": "Taesa", "SBSP3.SA": "Sabesp", "CMIG4.SA": "Cemig",
        "EQTL3.SA": "Equatorial",
    },
    "Saúde": {
        "RDOR3.SA": "Rede D'Or", "HAPV3.SA": "Hapvida", "FLRY3.SA": "Fleury",
    },
    "Exterior (exemplos)": {
        "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
        "GOOGL": "Alphabet (Google)", "KO": "Coca-Cola",
    },
}
NOMES = {tk: nm for g in CATALOGO.values() for tk, nm in g.items()}

# Só oferece o que está no CSV; mantém os grupos
CHOICES = {}
for grupo, d in CATALOGO.items():
    disp = {tk: f"{nm} ({tk.replace('.SA', '')})" for tk, nm in d.items() if tk in ATIVOS_DISP}
    if disp:
        CHOICES[grupo] = disp

DEFAULT = [a for a in ["HGRE11.SA", "BTLG11.SA", "HGRU11.SA", "VGIR11.SA"] if a in ATIVOS_DISP]
if not DEFAULT:
    DEFAULT = ATIVOS_DISP[:4]


def sid(ticker: str) -> str:
    return "w_" + ticker.replace(".", "_").replace("^", "_").replace("-", "_")


def nome_ativo(ticker: str) -> str:
    nm = NOMES.get(ticker)
    return f"{nm} ({ticker.replace('.SA', '')})" if nm else ticker


def loterias_inputs():
    """Radios das loterias (uma por decisão). A = seguro, B = arriscado."""
    itens = []
    for i, p in enumerate(LOTERIAS):
        itens.append(ui.input_radio_buttons(
            f"lot{i}", f"Decisão {i + 1}: chance alta = {p:.0%}",
            choices={
                "A": f"Seguro: {p:.0%} → R$ 40, senão R$ 32",
                "B": f"Arriscado: {p:.0%} → R$ 77, senão R$ 2",
            },
            selected="A",
        ))
    return itens


# ---------------------------------------------------------------- UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            "tickers", "Ativos (busque por nome ou código)",
            choices=CHOICES, selected=DEFAULT, multiple=True,
            options={"placeholder": "ex.: Petrobras, ITUB4, Vale..."},
        ),
        ui.output_ui("status_dados"),
        ui.hr(),
        ui.markdown("**Pesos** (0% remove o ativo; renormaliza para somar 100%)."),
        ui.output_ui("weight_sliders"),
        ui.input_action_button("igualar", "Igualar pesos", class_="btn-sm"),
        ui.hr(),
        ui.input_slider("rf", "Taxa livre de risco (% a.a.)", min=0, max=20, value=10, step=0.5),
        ui.input_slider("janela", "Janela (meses)", min=6, max=24,
                        value=min(24, max(6, len(RETORNOS) // 21)), step=3),
        title="Monte sua Carteira",
        width=330,
    ),
    ui.accordion(
        ui.accordion_panel(
            "1. Descubra o seu perfil de risco (loterias de Kahneman)",
            ui.markdown(
                "Em cada decisão, escolha entre uma aposta **segura** (A) e uma "
                "**arriscada** (B) com o mesmo padrão de chances. Quanto mais vezes você "
                "fica no seguro, mais **avesso ao risco** é o seu perfil. O app converte "
                "isso no coeficiente $\\lambda$ da utilidade $U = \\mu - \\tfrac{\\lambda}{2}\\sigma^2$ "
                "e marca, na fronteira, a carteira ótima para você."
            ),
            ui.layout_columns(*loterias_inputs(), col_widths=(4, 4, 4, 4, 4, 4)),
            ui.hr(),
            ui.markdown(
                "**Efeito de enquadramento (Kahneman e Tversky).** As duas decisões "
                "abaixo têm o mesmo valor esperado. Veja se você muda de atitude entre "
                "ganhos e perdas:"
            ),
            ui.layout_columns(
                ui.input_radio_buttons(
                    "frame_ganho", "Cenário de ganhos: você prefere",
                    choices={
                        "seguro": "Receber R$ 1.000 garantidos",
                        "aposta": "50% de ganhar R$ 2.000 (50% de nada)",
                    },
                    selected="seguro",
                ),
                ui.input_radio_buttons(
                    "frame_perda", "Cenário de perdas: você prefere",
                    choices={
                        "seguro": "Perder R$ 1.000 garantidos",
                        "aposta": "50% de perder R$ 2.000 (50% de nada)",
                    },
                    selected="aposta",
                ),
                col_widths=(6, 6),
            ),
            ui.output_ui("nota_frame"),
        ),
        open=True,
    ),
    ui.layout_columns(
        ui.value_box("Retorno (a.a.)", ui.output_text("vb_ret")),
        ui.value_box("Volatilidade (a.a.)", ui.output_text("vb_vol")),
        ui.value_box("Sharpe", ui.output_text("vb_sharpe")),
        ui.value_box("VaR 95% (dia)", ui.output_text("vb_var")),
        ui.value_box("Seu perfil (λ)", ui.output_text("vb_lambda")),
        fill=False,
    ),
    ui.layout_columns(
        ui.card(ui.card_header("Fronteira: sua carteira, 3000 aleatórias e a ótima do seu perfil"),
                output_widget("plot_fronteira"),
                ui.output_ui("txt_sugestao")),
        ui.card(ui.card_header("Pesos (renormalizados)"), output_widget("plot_pesos")),
        col_widths=(8, 4),
    ),
    ui.layout_columns(
        ui.card(ui.card_header("Retorno acumulado da carteira vs. Ibovespa"), output_widget("plot_backtest")),
        ui.card(ui.card_header("Correlação dos ativos na carteira"), output_widget("plot_corr")),
        col_widths=(7, 5),
    ),
    title="Lab 3: Monte sua Carteira",
    fillable=True,
)


# ---------------------------------------------------------------- server
def server(input, output, session):

    @reactive.calc
    def ativos_sel():
        return list(input.tickers())

    @render.ui
    def weight_sliders():
        ativos = ativos_sel()
        if not ativos:
            return ui.p(ui.tags.em("Escolha ao menos um ativo."))
        passo = max(5, round(100 / len(ativos) / 5) * 5)
        return ui.TagList(*[
            ui.input_slider(sid(a), nome_ativo(a), min=0, max=100, value=passo, step=5, post="%")
            for a in ativos
        ])

    @render.ui
    def status_dados():
        d0 = RETORNOS.index.min().date()
        d1 = RETORNOS.index.max().date()
        return ui.tags.small(
            f"{len(ATIVOS_DISP)} ativos no catálogo | dados de {d0} a {d1}",
            style="color:#666;",
        )

    @reactive.effect
    @reactive.event(input.igualar)
    def _igualar():
        ativos = ativos_sel()
        if ativos:
            v = max(5, round(100 / len(ativos) / 5) * 5)
            for a in ativos:
                ui.update_slider(sid(a), value=v)

    # ------------------------------------------------ perfil de risco (lambda)
    @reactive.calc
    def lam():
        n_safe = 0
        for i in range(len(LOTERIAS)):
            try:
                if input[f"lot{i}"]() == "A":
                    n_safe += 1
            except Exception:
                pass
        return LAMBDA_MAP[min(n_safe, 6)]

    @render.text
    def vb_lambda():
        lv = lam()
        return f"{lv:g} · {perfil_label(lv)}"

    @render.ui
    def nota_frame():
        try:
            g = input.frame_ganho()
            p = input.frame_perda()
        except Exception:
            return ui.div()
        if g == "seguro" and p == "aposta":
            txt = ("Você exibiu o **efeito reflexão** de Kahneman e Tversky: avesso ao "
                   "risco nos ganhos (prefere o certo) e propenso ao risco nas perdas "
                   "(arrisca para evitar a perda certa). É a marca da **aversão à perda**.")
            cor = INSPER_ROXO
        elif g == "aposta" and p == "seguro":
            txt = ("Atitude invertida em relação ao padrão usual: propenso nos ganhos e "
                   "avesso nas perdas. Vale discutir o que motivou cada escolha.")
            cor = INSPER_GRAY
        else:
            txt = ("Você manteve a mesma atitude (consistente) nos dois cenários. Boa parte "
                   "das pessoas troca de lado: esse é o ponto da teoria do prospecto.")
            cor = INSPER_GRAY
        return ui.div(ui.markdown(txt), style=f"border-left:4px solid {cor};padding-left:10px;")

    @reactive.calc
    def pesos():
        ativos = ativos_sel()
        if not ativos:
            return pd.Series(dtype=float)
        raw = []
        for a in ativos:
            try:
                v = input[sid(a)]()
            except Exception:
                v = None
            raw.append(0.0 if v is None else float(v))
        raw = np.array(raw, dtype=float)
        total = raw.sum()
        if total <= 0:
            return pd.Series(0.0, index=ativos)
        return pd.Series(raw / total, index=ativos)

    @reactive.calc
    def ret_janela():
        n = int(input.janela()) * 21
        return RETORNOS.tail(n)

    @reactive.calc
    def stats_carteira():
        w = pesos()
        ativos_w = list(w[w > 0].index) if len(w) else []
        d = ret_janela()
        if not ativos_w:
            return None
        da = d[ativos_w].dropna()
        if len(da) < 5:
            return None
        wv = w[ativos_w].values
        mean_ret = da.mean().values
        cov = da.cov().values
        port_ret_d = float(wv @ mean_ret)
        port_vol_d = float(np.sqrt(wv @ cov @ wv))
        rf_d = (input.rf() / 100) / 252
        sharpe = (port_ret_d - rf_d) / port_vol_d if port_vol_d > 0 else np.nan
        port_series = (da * wv).sum(axis=1)
        var95 = -np.percentile(port_series, 5)
        return dict(
            ret_a=port_ret_d * 252, vol_a=port_vol_d * np.sqrt(252),
            sharpe=sharpe * np.sqrt(252), var=var95,
            da=da, wv=wv, mean_ret=mean_ret, cov=cov, ativos=ativos_w,
        )

    @reactive.calc
    def frente():
        """Nuvem Monte Carlo das carteiras possíveis (retorno/risco anuais, em %)."""
        s = stats_carteira()
        if s is None or len(s["ativos"]) < 2:
            return None
        rng = np.random.default_rng(42)
        W = rng.dirichlet(np.ones(len(s["ativos"])), 3000)
        rr = (W @ s["mean_ret"]) * 252 * 100
        rk = np.sqrt(np.einsum("ij,jk,ik->i", W, s["cov"], W)) * np.sqrt(252) * 100
        return dict(W=W, rr=rr, rk=rk, sh=rr / rk, ativos=s["ativos"])

    @reactive.calc
    def sugestao():
        """Carteira que maximiza U = mu - lambda/2 * sigma^2 na nuvem (perfil do aluno)."""
        f = frente()
        if f is None:
            return None
        lv = lam()
        U = (f["rr"] / 100) - 0.5 * lv * ((f["rk"] / 100) ** 2)
        j = int(np.argmax(U))
        return dict(j=j, lam=lv, rr=float(f["rr"][j]), rk=float(f["rk"][j]),
                    pesos=pd.Series(f["W"][j], index=f["ativos"]))

    @render.text
    def vb_ret():
        s = stats_carteira()
        return "-" if s is None else f"{s['ret_a'] * 100:.1f}%"

    @render.text
    def vb_vol():
        s = stats_carteira()
        return "-" if s is None else f"{s['vol_a'] * 100:.1f}%"

    @render.text
    def vb_sharpe():
        s = stats_carteira()
        return "-" if s is None else f"{s['sharpe']:.2f}"

    @render.text
    def vb_var():
        s = stats_carteira()
        return "-" if s is None else f"{s['var'] * 100:.2f}%"

    @render.ui
    def txt_sugestao():
        sug = sugestao()
        if sug is None:
            return ui.div()
        pes = sug["pesos"]
        pes = pes[pes > 0.01].sort_values(ascending=False)
        itens = ", ".join(f"{a.replace('.SA', '')} {p:.0%}" for a, p in pes.items())
        return ui.markdown(
            f"**Carteira ótima para o seu perfil** (λ = {sug['lam']:g}, {perfil_label(sug['lam'])}): "
            f"retorno {sug['rr']:.1f}% a.a., risco {sug['rk']:.1f}% a.a. "
            f"Pesos sugeridos: {itens}."
        )

    @render_widget
    def plot_fronteira():
        f = frente()
        fig = go.Figure()
        if f is None:
            fig.add_annotation(text="Selecione ao menos 2 ativos com peso > 0.",
                               showarrow=False, font=dict(size=16))
            fig.update_layout(template="plotly_white")
            return fig
        s = stats_carteira()
        fig.add_trace(go.Scatter(
            x=f["rk"], y=f["rr"], mode="markers",
            marker=dict(size=4, color=f["sh"], colorscale="Viridis", opacity=0.55,
                        showscale=True, colorbar=dict(title="Sharpe")),
            name="Aleatórias", hovertemplate="risco %{x:.1f}%<br>retorno %{y:.1f}%<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[s["vol_a"] * 100], y=[s["ret_a"] * 100], mode="markers",
            marker=dict(size=20, color=INSPER_RED, symbol="star",
                        line=dict(width=1, color="white")),
            name="Sua carteira",
            hovertemplate="SUA CARTEIRA<br>risco %{x:.1f}%<br>retorno %{y:.1f}%<extra></extra>"))
        sug = sugestao()
        if sug is not None:
            fig.add_trace(go.Scatter(
                x=[sug["rk"]], y=[sug["rr"]], mode="markers",
                marker=dict(size=18, color=INSPER_AMARELO, symbol="diamond",
                            line=dict(width=1.5, color="black")),
                name=f"Ótima p/ seu perfil (λ={sug['lam']:g})",
                hovertemplate="ÓTIMA P/ SEU PERFIL<br>risco %{x:.1f}%<br>retorno %{y:.1f}%<extra></extra>"))
        fig.update_layout(template="plotly_white", xaxis_title="Risco anual (%)",
                          yaxis_title="Retorno anual (%)", margin=dict(t=10, r=10, l=10, b=10),
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig

    @render_widget
    def plot_pesos():
        w = pesos()
        w = w[w > 0] if len(w) else w
        if len(w) == 0:
            return go.Figure().update_layout(template="plotly_white")
        fig = px.bar(x=w.values, y=[nome_ativo(a) for a in w.index], orientation="h",
                     template="plotly_white", color_discrete_sequence=[INSPER_ROXO])
        fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Peso", yaxis_title=None,
                          margin=dict(t=10, r=10, l=10, b=10), showlegend=False)
        return fig

    @render_widget
    def plot_backtest():
        s = stats_carteira()
        fig = go.Figure()
        if s is None:
            return fig.update_layout(template="plotly_white")
        d = ret_janela()
        port = (d[s["ativos"]].dropna() * s["wv"]).sum(axis=1)
        traces = [(port, "Sua carteira", INSPER_RED)]
        if BENCH in d.columns:
            bench = d[BENCH].reindex(port.index).dropna()
            port = port.reindex(bench.index)
            traces = [(port, "Sua carteira", INSPER_RED), (bench, "Ibovespa", INSPER_GRAY)]
        for serie, nome, cor in traces:
            cum = (1 + serie).cumprod()
            fig.add_trace(go.Scatter(x=cum.index.strftime("%Y-%m-%d").tolist(),
                                     y=cum.values, name=nome, line=dict(color=cor, width=2)))
        fig.update_layout(template="plotly_white", yaxis_title="Retorno acumulado",
                          hovermode="x unified", margin=dict(t=10, r=10, l=10, b=10),
                          xaxis=dict(type="date"),
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        return fig

    @render_widget
    def plot_corr():
        s = stats_carteira()
        if s is None or len(s["ativos"]) < 2:
            return go.Figure().update_layout(template="plotly_white")
        corr = s["da"].corr()
        labels = [a.replace(".SA", "") for a in corr.columns]
        fig = px.imshow(corr.values, x=labels, y=labels, text_auto=".2f",
                        color_continuous_scale=[INSPER_TURQUESA, "#FFFFFF", INSPER_RED],
                        zmin=-1, zmax=1, template="plotly_white")
        fig.update_layout(margin=dict(t=10, r=10, l=10, b=10))
        return fig


app = App(app_ui, server)
