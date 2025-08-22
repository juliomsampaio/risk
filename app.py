Pular para o conteúdo
Histórico de chats

Você disse:
Imagem carregada
com base nessa planilha, em que uma estratégia ganha, em média, 12 reais a cada 1 real perdido e acerta 30% das vezes, com um capital inicial de 10000, crie um código em python para fazer simulações aleatórias de 10 mil situações possíveis de 50 operações seguidas, usando 2% do capital existente a cada operação, usando 5%, 10%, 15%, 20%, 25% e 30%. no final, preciso de saber todas as estatísticas sobre essas simulações
O ChatGPT disse:

Pensou por 24s

Analisado
Sempre exibir os detalhes

# Monte Carlo simulation for position sizing and asymmetric R/R
# - 10,000 simulations
# - 50 trades each
# - Win prob = 0.30
# - Win/Loss payoff in R: +12R when win, -1R when loss
# - Risk a fixed % of current equity per trade for each scenario:
#   2%, 5%, 10%, 15%, 20%, 25%, 30%
#
# Outputs:
# - Aggregated stats per risk level
# - CSV with the table for download

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import json
from caas_jupyter_tools import display_dataframe_to_user

# ----- Parameters -----
SEED = 42
np.random.seed(SEED)

start_equity = 10_000.0
win_prob = 0.30
r_win = 12.0
r_loss = -1.0
n_trades = 50
n_sims = 10_000
risk_levels = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ----- Helpers -----
def simulate_paths(risk_pct: float, n_sims: int, n_trades: int, start_equity: float,
                   win_prob: float, r_win: float, r_loss: float, seed: int = None
                  ) -> Dict[str, np.ndarray]:
    """
    Returns a dict of arrays per-simulation: final_equity, max_drawdown, wins, longest_loss_streak, longest_win_streak
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Pre-allocate
    final_equity = np.empty(n_sims, dtype=np.float64)
    max_drawdowns = np.empty(n_sims, dtype=np.float64)  # stored as negative percentages (e.g., -0.35)
    wins_count = np.empty(n_sims, dtype=np.int32)
    longest_loss_streak = np.empty(n_sims, dtype=np.int32)
    longest_win_streak = np.empty(n_sims, dtype=np.int32)

    # Simulate each path
    for s in range(n_sims):
        equity = start_equity
        peak = equity
        max_dd = 0.0
        wins = 0
        cur_loss_streak = 0
        cur_win_streak = 0
        best_loss_streak = 0
        best_win_streak = 0

        # simulate wins/losses in batch for speed
        outcomes = rng.random(n_trades) < win_prob  # True = win

        for o in outcomes:
            risk_amount = risk_pct * equity
            if o:  # win
                equity += risk_amount * r_win
                wins += 1
                cur_win_streak += 1
                cur_loss_streak = 0
                if cur_win_streak > best_win_streak:
                    best_win_streak = cur_win_streak
            else:  # loss
                equity += risk_amount * r_loss
                cur_loss_streak += 1
                cur_win_streak = 0
                if cur_loss_streak > best_loss_streak:
                    best_loss_streak = cur_loss_streak

            if equity > peak:
                peak = equity
            # drawdown (as negative fraction)
            dd = (equity / peak) - 1.0
            if dd < max_dd:
                max_dd = dd

        final_equity[s] = equity
        max_drawdowns[s] = max_dd
        wins_count[s] = wins
        longest_loss_streak[s] = best_loss_streak
        longest_win_streak[s] = best_win_streak

    return {
        "final_equity": final_equity,
        "max_drawdown": max_drawdowns,
        "wins": wins_count,
        "longest_loss_streak": longest_loss_streak,
        "longest_win_streak": longest_win_streak,
    }


def summarize_distribution(x: np.ndarray, percentiles: Tuple[float, ...] = (1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:
    out = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
    for p in percentiles:
        out[f"p{p:02.0f}"] = float(np.percentile(x, p))
    return out


def aggregate_stats(per_sim: Dict[str, np.ndarray], start_equity: float) -> Dict[str, float]:
    fe = per_sim["final_equity"]
    dd = per_sim["max_drawdown"]
    wins = per_sim["wins"]
    ll = per_sim["longest_loss_streak"]
    lw = per_sim["longest_win_streak"]

    # Final equity stats
    fe_stats = summarize_distribution(fe)

    # Max drawdown stats (report as positive % magnitudes)
    dd_magnitude = -dd  # convert to positive fractions
    dd_stats = summarize_distribution(dd_magnitude)

    # Other probabilities and counts
    prob_finish_below_start = float(np.mean(fe < start_equity))
    prob_dd_gt_50 = float(np.mean(dd <= -0.50))
    prob_dd_gt_70 = float(np.mean(dd <= -0.70))
    prob_dd_gt_80 = float(np.mean(dd <= -0.80))

    # wins / streaks
    wins_mean = float(np.mean(wins))
    wins_std = float(np.std(wins, ddof=1))
    ll_mean = float(np.mean(ll))
    ll_p95 = float(np.percentile(ll, 95))
    lw_mean = float(np.mean(lw))
    lw_p95 = float(np.percentile(lw, 95))

    # Merge a compact set of key metrics
    summary = {
        # Final equity (currency)
        "final_mean": fe_stats["mean"],
        "final_median": fe_stats["p50"],
        "final_p05": fe_stats["p05"],
        "final_p25": fe_stats["p25"],
        "final_p75": fe_stats["p75"],
        "final_p95": fe_stats["p95"],
        "final_min": fe_stats["min"],
        "final_max": fe_stats["max"],
        # Total return (%)
        "ret_mean_%": 100 * (fe_stats["mean"] / start_equity - 1),
        "ret_median_%": 100 * (fe_stats["p50"] / start_equity - 1),
        "ret_p05_%": 100 * (fe_stats["p05"] / start_equity - 1),
        "ret_p95_%": 100 * (fe_stats["p95"] / start_equity - 1),
        # Drawdown stats (positive % magnitudes)
        "mdd_mean_%": 100 * dd_stats["mean"],
        "mdd_median_%": 100 * dd_stats["p50"],
        "mdd_p95_%": 100 * dd_stats["p95"],
        "mdd_max_%": 100 * dd_stats["max"],
        # Risk metrics
        "prob_finish_below_start_%": 100 * prob_finish_below_start,
        "prob_dd>50%_%": 100 * prob_dd_gt_50,
        "prob_dd>70%_%": 100 * prob_dd_gt_70,
        "prob_dd>80%_%": 100 * prob_dd_gt_80,
        # Wins & streaks
        "mean_wins": wins_mean,
        "std_wins": wins_std,
        "mean_longest_loss_streak": ll_mean,
        "p95_longest_loss_streak": ll_p95,
        "mean_longest_win_streak": lw_mean,
        "p95_longest_win_streak": lw_p95,
    }
    return summary


# ----- Run simulations for all risk levels -----
rows = []
details_by_risk: Dict[float, Dict[str, np.ndarray]] = {}

for risk in risk_levels:
    per_sim = simulate_paths(
        risk_pct=risk,
        n_sims=n_sims,
        n_trades=n_trades,
        start_equity=start_equity,
        win_prob=win_prob,
        r_win=r_win,
        r_loss=r_loss,
        seed=SEED + int(risk*10000),  # vary seed per risk level
    )
    details_by_risk[risk] = per_sim
    stats = aggregate_stats(per_sim, start_equity)
    stats["risk_pct"] = risk
    rows.append(stats)

summary_df = pd.DataFrame(rows).set_index("risk_pct").sort_index()

# Order columns nicely
cols_order = [
    "final_mean","final_median","final_p05","final_p25","final_p75","final_p95","final_min","final_max",
    "ret_mean_%","ret_median_%","ret_p05_%","ret_p95_%",
    "mdd_mean_%","mdd_median_%","mdd_p95_%","mdd_max_%",
    "prob_finish_below_start_%","prob_dd>50%_%","prob_dd>70%_%","prob_dd>80%_%",
    "mean_wins","std_wins","mean_longest_loss_streak","p95_longest_loss_streak",
    "mean_longest_win_streak","p95_longest_win_streak",
]
summary_df = summary_df[cols_order]

# Save aggregated stats
csv_path = "/mnt/data/montecarlo_summary_stats.csv"
summary_df.to_csv(csv_path, float_format="%.4f")

# Show to user
display_dataframe_to_user("Monte Carlo Summary (10k sims, 50 trades)", summary_df)

csv_path
Resultado
'/mnt/data/montecarlo_summary_stats.csv'
Monte Carlo Summary (10k sims, 50 trades)


Here you go — I ran 10,000 Monte Carlo simulations of 50 trades for each risk level (2%, 5%, 10%, 15%, 20%, 25%, 30%) with:

Starting capital: R$10,000

Win rate: 30%

Payoff: +12R on wins, −1R on losses (R = % of current equity at risk each trade)

I used true compounding (risk is a fixed % of current equity each trade). For each scenario you get final-capital distribution stats, returns, max drawdown (as %), chance of finishing below the starting capital, probabilities of deep drawdowns (>50/70/80%), and win/streak metrics.

I displayed the full summary table to you. You can also grab it as a CSV:

Download the summary CSV

If you want, I can also show histograms of final capital or max drawdown per risk level, or save every path’s equity curve for deeper analysis. 

Você disse:

ajuste para usar no google colabs
O ChatGPT disse:
Perfeito 👍
Aqui está a versão ajustada para rodar diretamente no Google Colab. Já inclui os imports, execução, e salva o CSV no ambiente do Colab, com opção de baixar para o seu computador:

# Monte Carlo Simulation em Google Colab
# Estratégia: Win 12R com 30% de chance, Lose -1R com 70%
# 10.000 simulações, 50 operações, usando % do capital em risco
# Riscos: 2%, 5%, 10%, 15%, 20%, 25%, 30%

import numpy as np
import pandas as pd

# ----- Parâmetros -----
start_equity = 10_000.0
win_prob = 0.30
r_win = 12.0
r_loss = -1.0
n_trades = 50
n_sims = 10_000
risk_levels = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ----- Funções -----
def simulate_paths(risk_pct, n_sims, n_trades, start_equity, win_prob, r_win, r_loss):
    rng = np.random.default_rng()
    final_equity, max_dd, wins, ll_streak, lw_streak = [], [], [], [], []

    for _ in range(n_sims):
        equity, peak = start_equity, start_equity
        max_draw = 0.0
        wcount = 0
        cur_loss, cur_win, best_loss, best_win = 0, 0, 0, 0

        outcomes = rng.random(n_trades) < win_prob
        for o in outcomes:
            risk_amt = risk_pct * equity
            if o:  # vitória
                equity += risk_amt * r_win
                wcount += 1
                cur_win += 1
                cur_loss = 0
                best_win = max(best_win, cur_win)
            else:  # derrota
                equity += risk_amt * r_loss
                cur_loss += 1
                cur_win = 0
                best_loss = max(best_loss, cur_loss)

            peak = max(peak, equity)
            dd = equity / peak - 1
            max_draw = min(max_draw, dd)

        final_equity.append(equity)
        max_dd.append(max_draw)
        wins.append(wcount)
        ll_streak.append(best_loss)
        lw_streak.append(best_win)

    return {
        "final_equity": np.array(final_equity),
        "max_dd": np.array(max_dd),
        "wins": np.array(wins),
        "ll_streak": np.array(ll_streak),
        "lw_streak": np.array(lw_streak)
    }

def summarize_distribution(x):
    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "p05": np.percentile(x, 5),
        "p95": np.percentile(x, 95),
        "min": np.min(x),
        "max": np.max(x)
    }

def aggregate_stats(sim, start_equity):
    fe = sim["final_equity"]
    dd = -sim["max_dd"]  # drawdowns em %
    stats = {}

    fe_stats = summarize_distribution(fe)
    dd_stats = summarize_distribution(dd)

    stats.update({
        "final_mean": fe_stats["mean"],
        "final_median": fe_stats["median"],
        "final_p05": fe_stats["p05"],
        "final_p95": fe_stats["p95"],
        "ret_mean_%": 100*(fe_stats["mean"]/start_equity - 1),
        "ret_median_%": 100*(fe_stats["median"]/start_equity - 1),
        "mdd_mean_%": 100*dd_stats["mean"],
        "mdd_p95_%": 100*dd_stats["p95"],
        "prob_below_start_%": 100*np.mean(fe < start_equity),
        "prob_dd>50%_%": 100*np.mean(dd > 0.5),
        "prob_dd>70%_%": 100*np.mean(dd > 0.7),
        "mean_wins": np.mean(sim["wins"]),
        "p95_loss_streak": np.percentile(sim["ll_streak"],95),
        "p95_win_streak": np.percentile(sim["lw_streak"],95)
    })
    return stats

# ----- Rodar simulações -----
rows = []
for r in risk_levels:
    sim = simulate_paths(r, n_sims, n_trades, start_equity, win_prob, r_win, r_loss)
    stats = aggregate_stats(sim, start_equity)
    stats["risk_%"] = int(r*100)
    rows.append(stats)

df = pd.DataFrame(rows).set_index("risk_%")
df
Como salvar o resultado no Colab
Se quiser baixar como CSV:

df.to_csv("montecarlo_summary.csv", index=True)

from google.colab import files
files.download("montecarlo_summary.csv")
👉 Quer que eu já adicione plots (histogramas/distribuições) para visualizar os resultados de cada nível de risco?

Você disse:
não entendi como foi feito, os valores deram muito altos. me explique seu raciocínio com calma
O ChatGPT disse:

Pensou por 2m 18s
Totally fair question — the numbers look “too big” at first sight.
Let me walk you (slowly) through exactly what I simulated and why the results explode.

1) What each trade does (the core math)
You risk a fixed fraction r of your current equity each trade (fractional Kelly-style compounding).

Payoff in R: +12R if you win, −1R if you lose.

With risk r per trade:

Win ⇒ equity multiplies by (1 + 12r)

Loss ⇒ equity multiplies by (1 − r)

Example for r = 2%:

Win ⇒ ×(1 + 12×0.02) = ×1.24 (gain 24%)

Loss ⇒ ×(1 − 0.02) = ×0.98 (lose 2%)

This is compounding: the next bet size is a % of the new equity.

2) Your edge is enormous
Hit rate p = 30% with payoff +12 / −1 gives a very large edge.

Expected additive return per trade (in “R”):
E
[
Δ
E
/
E
]
=
p
⋅
12
r
+
(
1
−
p
)
⋅
(
−
r
)
=
r
 
(
13
p
−
1
)
E[ΔE/E]=p⋅12r+(1−p)⋅(−r)=r(13p−1)
With 
p
=
0.30
⇒
13
p
−
1
=
2.9
p=0.30⇒13p−1=2.9.
So expected % gain per trade ≈ 
2.9
 
r
2.9r.

Examples:

r = 2% ⇒ ~+5.8% expected per trade

r = 10% ⇒ ~+29% expected per trade

r = 30% ⇒ ~+87% expected per trade

That is huge. Even at 2% risk, +5.8% per trade compounded over many trades is explosive.

3) What this implies over 50 trades
Let 
G
G be the per-trade growth factor: win → 
1
+
12
r
1+12r, loss → 
1
−
r
1−r.
The expected per-trade factor is:

E
[
G
]
=
p
(
1
+
12
r
)
+
(
1
−
p
)
(
1
−
r
)
=
1
+
r
(
13
p
−
1
)
=
1
+
2.9
r
E[G]=p(1+12r)+(1−p)(1−r)=1+r(13p−1)=1+2.9r
If trades are independent, the expected final equity after 50 trades is:

E
final
≈
E
0
⋅
(
1
+
2.9
r
)
50
E 
final
​
 ≈E 
0
​
 ⋅(1+2.9r) 
50
 
r = 2% ⇒ 
(
1.058
)
50
≈
16.4
(1.058) 
50
 ≈16.4 ⇒ ≈ 10k × 16.4 = 164k (expectation)

r = 5% ⇒ 
(
1.145
)
50
(1.145) 
50
  is enormous (hundreds of ×).
(The “expectation” is dominated by a few extreme lucky paths.)

This is why the averages in Monte Carlo look “very high”.

Important: Expectation ≠ typical outcome. Because of compounding and skew, the mean is pulled up by rare, gigantic winners.

4) What’s “typical” (median) growth?
A better “typical” measure is the median, tied to the average log-growth:

E
[
ln
⁡
G
]
=
p
ln
⁡
(
1
+
12
r
)
+
(
1
−
p
)
ln
⁡
(
1
−
r
)
E[lnG]=pln(1+12r)+(1−p)ln(1−r)
Median per trade ≈ 
exp
⁡
(
E
[
ln
⁡
G
]
)
exp(E[lnG]). Over 50 trades:

median factor
≈
exp
⁡
 ⁣
(
50
⋅
E
[
ln
⁡
G
]
)
median factor≈exp(50⋅E[lnG])
For r = 2%:

E
[
ln
⁡
G
]
≈
0.3
ln
⁡
(
1.24
)
+
0.7
ln
⁡
(
0.98
)
≈
0.050
E[lnG]≈0.3ln(1.24)+0.7ln(0.98)≈0.050

Median per trade ≈ 
e
0.050
≈
1.0517
e 
0.050
 ≈1.0517 (≈ +5.17%/trade)

Over 50 trades: 
1.0517
50
≈
12
×
1.0517 
50
 ≈12×
⇒ typical final ≈ R$120k (still big, but lower than the average ~R$164k).

As you increase r, the median also becomes huge, because a single win is massive:

r = 15% ⇒ Win multiplies equity by 2.8×

r = 30% ⇒ Win multiplies equity by 4.6×
Two or three wins anywhere in the path can blow up equity.

5) Why your sheet and my sim may disagree
From your screenshot, I see columns with 2%, 5%, 10%, … and final capitals like R$21,600 for 2%. That looks like a specific fixed sequence of wins/losses (not 50 trades, and not random) — and possibly fewer trades — which can absolutely end much smaller.

My code:

runs 10,000 random paths

each path has 50 trades

risk is a % of current equity (true compounding)

payoff is +12R / −1R with p = 30%

Given that setup, large outcomes are mathematically expected.

If you want results closer to your sheet:

use the exact same number of trades your sheet uses,

use a fixed order of wins/losses (not random),

or use fixed-dollar risk (2% of the initial 10k, every trade), not a % of current equity.

6) Risk of deep drawdowns (very real)
Despite the huge upside, losing streaks hurt:

With r = 10%, 10 losses in a row ⇒ equity × 
0.9
10
≈
0.35
0.9 
10
 ≈0.35 (−65% drawdown).

In 50 trades with 70% loss probability, long loss streaks are not rare.

The Monte Carlo I gave reports probabilities of >50%, >70%, >80% drawdown — these can be substantial at high r.

This connects to Kelly: for an edge with b = 12 (profit multiple) and p = 0.3, the Kelly fraction is

f
∗
=
b
p
−
(
1
−
p
)
b
=
12
⋅
0.3
−
0.7
12
=
2.9
12
≈
24.2
%
f 
∗
 = 
b
bp−(1−p)
​
 = 
12
12⋅0.3−0.7
​
 = 
12
2.9
​
 ≈24.2%
Betting above ~24% tends to maximize growth but also explodes volatility and tail risk.

Many pros use fractional Kelly (e.g., ½-Kelly ≈ 12%) to control drawdowns.

7) Quick 3-trade example (r = 2%)
Start R$10,000; outcomes: Loss → Win → Loss

After loss: 10,000 × 0.98 = 9,800

Then win: 9,800 × 1.24 = 12,152

Then loss: 12,152 × 0.98 = 11,909
One single win more than undoes several small losses because it’s +24% vs −2%.

Bottom line
Your parameters (30% hit, +12/−1) imply an extremely strong edge.

When you risk a % of current equity, compounding + the huge +12R wins make the distribution very skewed with very large averages and high medians — especially by 50 trades.

The code is doing exactly that model; the “very high” results are a direct consequence of those assumptions.

If you tell me which modeling you’d prefer, I’ll adjust the notebook:

Fixed-dollar risk (2% of the initial 10k every trade — no compounding)

Cap the win (e.g., average 12R but with variability or maximum)

Include costs/slippage

Use your exact win/loss sequence from the sheet

Different number of trades (e.g., 20 instead of 50)

Say the word, and I’ll deliver that version.

Você disse:
reescreva em português
O ChatGPT disse:
Beleza 🙌 Vou reexplicar tudo em português, com calma, para você entender por que os valores da simulação ficaram tão altos.

1) O que acontece em cada operação
Você arrisca uma fração fixa r do seu capital atual em cada operação (isso é importante: risco sobre o capital atualizado, não sobre o capital inicial).

O payoff está em R:

+12R quando ganha

−1R quando perde

Assim, se o risco é r:

Vitória → capital × (1 + 12r)

Derrota → capital × (1 − r)

🔹 Exemplo com r = 2%:

Vitória = multiplicar por (1 + 0,24) = 1,24 (ganha 24%)

Derrota = multiplicar por (1 − 0,02) = 0,98 (perde 2%)

E isso vai se acumulando (composto).

2) A vantagem da estratégia é gigantesca
Probabilidade de acerto 30%, payoff +12/−1, dá uma expectativa positiva enorme.

Valor esperado por operação (em % do capital):
E
[
Δ
E
/
E
]
=
r
⋅
(
13
p
−
1
)
E[ΔE/E]=r⋅(13p−1)
Com 
p
=
0
,
30
⇒
13
p
−
1
=
2
,
9
p=0,30⇒13p−1=2,9

Ou seja, lucro esperado ≈ 2,9 × r por trade.

Exemplos:

r = 2% ⇒ ganho esperado ≈ +5,8% por operação

r = 10% ⇒ ganho esperado ≈ +29% por operação

r = 30% ⇒ ganho esperado ≈ +87% por operação (!!)

Isso é muito acima de qualquer estratégia “normal”.

3) O que acontece depois de 50 operações
O fator de crescimento médio por trade é:

E
[
G
]
=
1
+
2
,
9
r
E[G]=1+2,9r
Depois de 50 trades:

E
final
≈
E
0
⋅
(
1
+
2
,
9
r
)
50
E 
final
​
 ≈E 
0
​
 ⋅(1+2,9r) 
50
 
r = 2% ⇒ (1,058)⁵⁰ ≈ 16,4 ⇒ 10.000 → 164.000 (esperado)

r = 5% ⇒ (1,145)⁵⁰ ≈ centenas de vezes o capital

r = 10% ⇒ (1,29)⁵⁰ ⇒ astronômico

⚠️ Isso mostra o motivo dos números da simulação: o valor esperado explode.

4) Mediana vs. Média (o que é “típico”)
O problema: o valor esperado é puxado para cima por poucos casos extremamente sortudos.
O que representa melhor o resultado típico é a mediana, que vem do log-growth:

E
[
ln
⁡
G
]
=
p
ln
⁡
(
1
+
12
r
)
+
(
1
−
p
)
ln
⁡
(
1
−
r
)
E[lnG]=pln(1+12r)+(1−p)ln(1−r)
Para r = 2%:

E[ln G] ≈ 0,050

Crescimento típico ≈ e^0,050 = +5,17% por operação

Em 50 trades ⇒ ≈ 12× o capital (120k)

Ainda é grande, mas menor que os 164k da média.

5) Por que sua planilha mostrou números bem menores
Na planilha que você me mandou:

parece que você usou um número fixo e pequeno de trades (talvez 12, não 50),

e com uma sequência específica de vitórias/derrotas,

e talvez sem compounding real (ou seja, arriscando sempre o mesmo valor fixo, não % do capital atualizado).

Isso faz o capital crescer bem menos do que no meu modelo de Monte Carlo.

6) Risco de grandes perdas
Mesmo com vantagem enorme, existem riscos:

Com r = 10%, 10 derrotas seguidas ⇒ capital × (0,9)¹⁰ ≈ 0,35 (−65% drawdown).

Em 50 trades com 70% de chance de perda, sequências longas de derrotas são prováveis.

Por isso a simulação também mede probabilidade de drawdown >50%, >70%, etc.

7) Fórmula de Kelly
Essa situação tem uma “aposta ótima” (Kelly):

f
∗
=
b
p
−
(
1
−
p
)
b
f 
∗
 = 
b
bp−(1−p)
​
 
onde b = 12, p = 0,3.

f
∗
=
12
×
0
,
3
−
0
,
7
12
≈
24
,
2
%
f 
∗
 = 
12
12×0,3−0,7
​
 ≈24,2%
Ou seja:

Apostar mais de 24% do capital por trade maximiza crescimento esperado,

mas aumenta o risco de ruína.

Profissionais geralmente usam meio Kelly (≈ 12%) para reduzir drawdowns.

8) Mini-exemplo com r = 2% e 3 trades (Perda → Vitória → Perda)
Inicial: 10.000

Perde: 10.000 × 0,98 = 9.800

Ganha: 9.800 × 1,24 = 12.152

Perde: 12.152 × 0,98 = 11.909

Ou seja: uma vitória de +24% compensa várias derrotas de −2%.

✅ Conclusão
Os valores “muito altos” não são erro do código: eles são consequência direta da enorme assimetria da estratégia (+12/−1, 30%).

No modelo de simulação (50 trades, risco % do capital, compounding real), é natural que o capital final chegue a centenas de milhares.

Sua planilha mostra valores menores porque:

Provavelmente usou menos trades,

Usou uma sequência fixa de vitórias/derrotas,

Possivelmente aplicou risco fixo em dinheiro, não % do capital atualizado.

👉 Quer que eu ajuste o código para simular sem compounding (risco fixo em reais por operação) para bater com o estilo da sua planilha?

Você disse:
acho que não fui claro, vou falar o que eu queria e você me fala se você fez assim ou diferente. esse valor de 12 é o fator de ganho que o trading view mostra na execução de backtest de estratégia
O ChatGPT disse:

Pensou por 13s
Obrigado por esclarecer! Então vamos alinhar os conceitos:

O que eu fiz antes
Eu interpretei o “12” como payoff por trade: cada operação vencedora ganha +12R e cada perdedora perde −1R (onde R é a fração do capital arriscada).
Com 30% de acerto, isso gera um edge gigantesco e, com alocação percentual do capital (compounding), os resultados realmente “explodem”.

O que o TradingView chama de “Fator de ganho”
No TradingView, o Fator de ganho (“Profit Factor”) é:

PF
=
Lucro Bruto
Preju
ı
ˊ
zo Bruto
PF= 
Preju 
ı
ˊ
 zo Bruto
Lucro Bruto
​
 
Ele não diz diretamente “ganha 12 por 1 perdido em cada trade**, e sim que a soma dos ganhos foi 12 vezes a soma das perdas no período do backtest.

Se quisermos traduzir PF para um payoff médio por trade (com o mesmo tamanho de risco por trade), usamos:

PF
=
p
⋅
W
‾
(
1
−
p
)
⋅
L
‾
⇒
W
‾
=
PF
⋅
(
1
−
p
)
p
⋅
L
‾
PF= 
(1−p)⋅ 
L
 
p⋅ 
W
 
​
 ⇒ 
W
 =PF⋅ 
p
(1−p)
​
 ⋅ 
L
 
onde:

p
p = taxa de acerto (0,30)

W
‾
W
  = ganho médio por trade vencedor (em múltiplos de R)

L
‾
L
  = perda média por trade perdedor (se fixarmos 
L
‾
=
1
R
L
 =1R)

Com PF = 12 e p = 0,30:

W
‾
=
12
⋅
0,70
0,30
⋅
1
R
=
28
R
W
 =12⋅ 
0,30
0,70
​
 ⋅1R=28R
Ou seja: para PF=12 com 30% de acerto, o ganho médio por trade vencedor seria 28R, não 12R.

Conclusão direta
Se o “12” que você citou é o Profit Factor do TradingView, eu simulei diferente do que você tinha em mente.

Para replicar um cenário com PF=12 e 30% de acerto, o modelo equivalente (com perdas = 1R) precisa usar vitórias médias = 28R (podemos usar 28R constante ou uma distribuição com média 28R).

O que você prefere que eu simule?
Posso te entregar, já em Colab, as duas versões:

Modelo “payoff fixo por trade” (o que eu fiz antes, mas você pode escolher o payoff):

Wins = +12R (ou +28R, se quisermos refletir PF=12 com p=30%)

Losses = −1R

Risco = % do capital corrente (compounding)

10.000 simulações × 50 trades

Modelo “alinha com Profit Factor” (mais realista para PF de backtest):

Definimos 
p
p, PF, 
L
‾
=
1
R
L
 =1R.

Calculamos 
W
‾
=
PF
⋅
1
−
p
p
W
 =PF⋅ 
p
1−p
​
 .

Podemos sortear o ganho vencedor em torno de 28R (ex.: lognormal com média 28R) para refletir variação entre trades, mantendo a média compatível com PF=12.

Risco = % do capital corrente.

10.000 simulações × 50 trades.

Diga qual você quer (ou ambos), que eu já te mando o notebook Colab prontinho conforme a definição correta do “12” que você deseja usar.

Você disse:
refaça usando o modelo alinhado com profit factor
O ChatGPT disse:

Pensou por 15s
Perfeito — abaixo vai um notebook Colab-ready que refaz as simulações alinhadas ao Profit Factor (PF).

Modelo adotado (alinhado ao PF)
PF = 12 (do TradingView) e taxa de acerto p = 0,30.

Definimos a perda média do trade perdedor como 1R.

Para que o PF seja respeitado, o ganho médio do trade vencedor precisa ser:

W
‾
  
=
  
PF
⋅
1
−
p
p
⋅
L
‾
  
=
  
12
⋅
0,7
0,3
⋅
1
  
=
  
28
R
W
 =PF⋅ 
p
1−p
​
 ⋅ 
L
 =12⋅ 
0,3
0,7
​
 ⋅1=28R
Default do código: payoff vencedor constante = +28R e perdedor −1R, garantindo PF≈12 no agregado.

(Opcional) Você pode ativar variabilidade do payoff vencedor (lognormal com média 28R), mantendo o PF-alvo no valor esperado.

Regras da simulação
Capital inicial: R$10.000

10.000 simulações independentes

50 operações por simulação

Risco por trade = % do capital atual (compounding real)

Níveis de risco: 2%, 5%, 10%, 15%, 20%, 25%, 30%

Estatísticas: distribuição do capital final, retornos, drawdown máx., probabilidades de DD>50/70/80%, prob. de terminar abaixo do inicial, nº de vitórias, maiores sequências de perdas/ganhos etc.

Código (cole no Google Colab e rode)
# Monte Carlo alinhado ao Profit Factor (TradingView)
# PF = 12, taxa de acerto p = 0.30
# Perda média por trade perdedor = 1R
# Logo, ganho médio por trade vencedor = 28R (constante por default)
#
# 10.000 simulações, 50 trades, risco = % do capital atual (compounding)
# Riscos: 2%, 5%, 10%, 15%, 20%, 25%, 30%

import numpy as np
import pandas as pd

# -----------------------------
# Parâmetros gerais
# -----------------------------
SEED = 42
rng = np.random.default_rng(SEED)

start_equity = 10_000.0
p_win = 0.30
profit_factor = 12.0

# Definição do payoff em "R"
loss_R = 1.0  # perda média por trade perdedor (em R)
win_mean_R = profit_factor * (1 - p_win) / p_win * loss_R  # = 28R

# Toggle: payoff vencedor constante (28R) OU lognormal com média 28R
USE_LOGNORMAL_WINS = False
# se usar lognormal, defina o desvio-padrão do log (sigma); média sempre recalibrada para 28R
LOGNORMAL_SIGMA = 0.6

n_trades = 50
n_sims = 10_000
risk_levels = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # fração do capital atual

# -----------------------------
# Funções auxiliares
# -----------------------------
def sample_win_payoff_R(size, rng):
    """Retorna vetores de payoffs vencedores em R.
       - Constante (= 28R) por default
       - Opcional: Lognormal com média 28R (mantendo PF no esperado)
    """
    if not USE_LOGNORMAL_WINS:
        return np.full(size, win_mean_R, dtype=float)
    # Lognormal: queremos média = win_mean_R
    # para lognormal: mean = exp(mu + 0.5*sigma^2) => mu = ln(mean) - 0.5*sigma^2
    mu = np.log(win_mean_R) - 0.5 * (LOGNORMAL_SIGMA**2)
    return rng.lognormal(mean=mu, sigma=LOGNORMAL_SIGMA, size=size)

def simulate_paths(risk_pct, n_sims, n_trades, start_equity, p_win, rng):
    """
    Simula n_sims trajetórias com n_trades cada, arriscando risk_pct do capital atual.
    Payoff vencedor: +win_mean_R*R (ou amostrado lognormal, média 28R). Perdas: -1R.
    Retorna dict com vetores por simulação.
    """
    final_equity = np.empty(n_sims, dtype=np.float64)
    max_drawdown = np.empty(n_sims, dtype=np.float64)  # negativo (ex: -0.35)
    wins = np.empty(n_sims, dtype=np.int32)
    longest_loss_streak = np.empty(n_sims, dtype=np.int32)
    longest_win_streak = np.empty(n_sims, dtype=np.int32)

    for s in range(n_sims):
        equity = start_equity
        peak = equity
        max_dd = 0.0
        wcount = 0
        cur_loss = cur_win = 0
        best_loss = best_win = 0

        # desenha resultados: True = win
        outcomes = rng.random(n_trades) < p_win

        # se wins forem variáveis, amostramos todos os payoffs vencedores desta trajetória
        win_payoffs = sample_win_payoff_R(n_trades, rng) if USE_LOGNORMAL_WINS else None

        for i, o in enumerate(outcomes):
            risk_amount = risk_pct * equity  # risco em R$
            if o:
                # payoff vencedor em R$
                payoff_R = win_payoffs[i] if USE_LOGNORMAL_WINS else win_mean_R
                equity += risk_amount * payoff_R
                wcount += 1
                cur_win += 1
                cur_loss = 0
                if cur_win > best_win:
                    best_win = cur_win
            else:
                equity += -risk_amount * loss_R
                cur_loss += 1
                cur_win = 0
                if cur_loss > best_loss:
                    best_loss = cur_loss

            if equity > peak:
                peak = equity
            dd = equity / peak - 1.0
            if dd < max_dd:
                max_dd = dd

        final_equity[s] = equity
        max_drawdown[s] = max_dd
        wins[s] = wcount
        longest_loss_streak[s] = best_loss
        longest_win_streak[s] = best_win

    return {
        "final_equity": final_equity,
        "max_drawdown": max_drawdown,  # negativo
        "wins": wins,
        "longest_loss_streak": longest_loss_streak,
        "longest_win_streak": longest_win_streak,
    }

def summarize_distribution(x, percentiles=(1,5,25,50,75,95,99)):
    x = np.asarray(x, dtype=float)
    out = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
    for p in percentiles:
        out[f"p{int(p):02d}"] = float(np.percentile(x, p))
    return out

def aggregate_stats(per_sim, start_equity):
    fe = per_sim["final_equity"]
    dd = per_sim["max_drawdown"]  # negativo
    wins = per_sim["wins"]
    ll = per_sim["longest_loss_streak"]
    lw = per_sim["longest_win_streak"]

    fe_stats = summarize_distribution(fe)
    dd_stats = summarize_distribution(-dd)  # magnitude positiva

    summary = {
        # Capital final (R$)
        "final_mean": fe_stats["mean"],
        "final_median": fe_stats["p50"],
        "final_p05": fe_stats["p05"],
        "final_p25": fe_stats["p25"],
        "final_p75": fe_stats["p75"],
        "final_p95": fe_stats["p95"],
        "final_min": fe_stats["min"],
        "final_max": fe_stats["max"],
        # Retornos (%)
        "ret_mean_%": 100 * (fe_stats["mean"]/start_equity - 1),
        "ret_median_%": 100 * (fe_stats["p50"]/start_equity - 1),
        "ret_p05_%": 100 * (fe_stats["p05"]/start_equity - 1),
        "ret_p95_%": 100 * (fe_stats["p95"]/start_equity - 1),
        # Drawdown (em %)
        "mdd_mean_%": 100 * dd_stats["mean"],
        "mdd_median_%": 100 * dd_stats["p50"],
        "mdd_p95_%": 100 * dd_stats["p95"],
        "mdd_max_%": 100 * dd_stats["max"],
        # Probabilidades
        "prob_finish_below_start_%": 100 * float(np.mean(fe < start_equity)),
        "prob_dd>50%_%": 100 * float(np.mean(dd <= -0.50)),
        "prob_dd>70%_%": 100 * float(np.mean(dd <= -0.70)),
        "prob_dd>80%_%": 100 * float(np.mean(dd <= -0.80)),
        # Vitórias e sequências
        "mean_wins": float(np.mean(wins)),
        "std_wins": float(np.std(wins, ddof=1)),
        "mean_longest_loss_streak": float(np.mean(ll)),
        "p95_longest_loss_streak": float(np.percentile(ll, 95)),
        "mean_longest_win_streak": float(np.mean(lw)),
        "p95_longest_win_streak": float(np.percentile(lw, 95)),
    }
    return summary

# -----------------------------
# Rodar todas as combinações de risco
# -----------------------------
rows = []
for r in risk_levels:
    # variar um pouco a semente por nível de risco para independência
    per_sim = simulate_paths(
        risk_pct=r,
        n_sims=n_sims,
        n_trades=n_trades,
        start_equity=start_equity,
        p_win=p_win,
        rng=np.random.default_rng(SEED + int(r*10000))
    )
    stats = aggregate_stats(per_sim, start_equity)
    stats["risk_pct"] = r
    rows.append(stats)

summary_df = pd.DataFrame(rows).set_index("risk_pct").sort_index()
summary_df
Salvar e baixar o CSV no Colab
summary_df.to_csv("montecarlo_pf12_summary.csv", float_format="%.6f")
from google.colab import files
files.download("montecarlo_pf12_summary.csv")
Observações importantes
Com PF=12 e p=30%, o ganho médio por trade vencedor precisa ser 28R para refletir fielmente o PF (assumindo perda média = 1R).

Como o risco é % do capital atual, o modelo é composto; os resultados podem ser grandes (média e mediana), mas o código também reporta drawdowns e probabilidades de cenários ruins.

Se quiser variabilidade nos ganhos vencedores, mude USE_LOGNORMAL_WINS = True e ajuste LOGNORMAL_SIGMA. A média permanece 28R (mantendo o PF no valor esperado), mas a dispersão aumenta.

Se quiser, eu acrescento histogramas (capital final, MDD) e gráficos das distribuições por nível de risco.

Você disse:
na verdade, vamos fazer diferente agora. eu quero analisar só o risco. simule aleatoriamente operações ganhadoras e perdedoras como na imagem que enviei anteriormente, com base na probabilidade de erro ou acerto. considere que sempre será investido um percentual com base no capital inicial (não será com base no valor atual mais). eu preciso de saber quantas vezes o capital foi zerado, para cada nível de risco, nessas 10 mil simulações de 50 operações sequenciais
O ChatGPT disse:

Pensou por 21s
Risco de Ruína (aposta fixa, PF=12, p=0.30)


Pronto — rodei as 10.000 simulações (50 operações) para cada nível de risco apostando sempre a mesma fração do capital inicial (sem redimensionar pela variação do saldo). Considerei:

p(acerto) = 30%

Profit Factor = 12 (com perda média = 1R ⇒ ganho médio do trade vencedor = 28R)

Em cada trade, o valor em risco é risk% × capital_inicial (fixo por toda a simulação)

O capital é atualizado a cada trade (lucros e perdas); contamos “zerou” quando o saldo ficou ≤ 0 em qualquer momento da sequência

Mostrei a tabela com bust_count e bust_rate_% por nível de risco.
Você também pode baixar:

Baixar CSV dos resultados

Quer que eu replique com outra probabilidade de acerto ou com PF diferente? Ou que eu compare com um cenário “neutro” (vitória = +1R, perda = −1R) só para isolar o efeito do risco puro? 

Você disse:
faça o código em python para eu rodar no google colab
O ChatGPT disse:
Perfeito — abaixo está um notebook pronto para o Google Colab. Ele simula 10.000 cenários com 50 operações, apostando sempre a mesma fração do capital inicial (sem redimensionar pelo saldo), calcula quantas vezes o capital zera para cada nível de risco e gera um CSV para download.

Modelo: p(acerto)=0,30, Profit Factor (PF)=12
Assumindo perda média = 1R, o ganho médio do trade vencedor é 28R para manter PF=12.
Em cada trade, o valor em risco é risk% × capital_inicial (fixo durante toda a simulação).

# Monte Carlo de Risco de Ruína (Google Colab)
# - Tamanho da aposta fixo em R$ (percentual do capital inicial, sem compounding de size)
# - Capital é atualizado a cada trade; ruína ocorre se equity <= 0 em qualquer momento
# - p_win = 0.30, PF = 12  => perda média = 1R, ganho médio vencedor = 28R
# - 10.000 simulações, 50 trades, riscos: 2%, 5%, 10%, 15%, 20%, 25%, 30%

import numpy as np
import pandas as pd

# ======================
# Parâmetros principais
# ======================
START_EQUITY = 10_000.0
P_WIN = 0.30
PROFIT_FACTOR = 12.0
LOSS_R = 1.0  # perda média do trade perdedor em R
WIN_MEAN_R = PROFIT_FACTOR * (1 - P_WIN) / P_WIN * LOSS_R  # = 28R

N_TRADES = 50
N_SIMS = 10_000
RISK_LEVELS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Reprodutibilidade
SEED = 123
rng = np.random.default_rng(SEED)

# ======================
# Opções de payoff vencedor
# ======================
# Por padrão, usamos payoff vencedor constante = WIN_MEAN_R para refletir exatamente PF=12 no esperado.
USE_LOGNORMAL_WINS = False  # mude para True se quiser variabilidade nos ganhos vencedores
LOGNORMAL_SIGMA = 0.6       # desvio-padrão do log (se usar lognormal)

def sample_win_payoff_R(size, rng):
    """Retorna payoffs vencedores (em R) para 'size' operações.
       - Se USE_LOGNORMAL_WINS=False, retorna constante WIN_MEAN_R.
       - Se True, retorna lognormal com média exatamente WIN_MEAN_R.
    """
    if not USE_LOGNORMAL_WINS:
        return np.full(size, WIN_MEAN_R, dtype=float)
    mu = np.log(WIN_MEAN_R) - 0.5 * (LOGNORMAL_SIGMA**2)
    return rng.lognormal(mean=mu, sigma=LOGNORMAL_SIGMA, size=size)

def simulate_busts_for_risk(risk_pct: float,
                            n_sims: int,
                            n_trades: int,
                            start_equity: float,
                            p_win: float,
                            rng: np.random.Generator) -> int:
    """Simula n_sims trajetórias de n_trades.
       Tamanho da aposta = risk_pct * start_equity (fixo).
       Retorna o número de simulações que 'quebram' (equity <= 0 em algum momento).
    """
    busts = 0
    bet_amount = risk_pct * start_equity  # aposta fixa em R$ por trade

    for _ in range(n_sims):
        equity = start_equity
        outcomes = rng.random(n_trades) < p_win  # True = win
        win_Rs = sample_win_payoff_R(n_trades, rng)

        busted = False
        for i, win in enumerate(outcomes):
            if win:
                equity += bet_amount * win_Rs[i]   # ganho = R$ apostado * payoff vencedor em R
            else:
                equity -= bet_amount * LOSS_R      # perda = R$ apostado * 1R

            if equity <= 0:
                busted = True
                break

        if busted:
            busts += 1

    return busts

# ======================
# Rodar simulações
# ======================
rows = []
for r in RISK_LEVELS:
    # usa semente diferente por nível de risco para independência entre cenários
    busts = simulate_busts_for_risk(
        risk_pct=r,
        n_sims=N_SIMS,
        n_trades=N_TRADES,
        start_equity=START_EQUITY,
        p_win=P_WIN,
        rng=np.random.default_rng(SEED + int(r*10_000))
    )
    rows.append({
        "risk_%": int(round(r * 100)),
        "bust_count": busts,
        "bust_rate_%": 100.0 * busts / N_SIMS
    })

result_df = pd.DataFrame(rows).set_index("risk_%").sort_index()
result_df
Baixar o CSV no Colab
# Salvar e baixar
result_df.to_csv("risk_of_ruin_fixed_bet_pf12.csv", float_format="%.6f")
from google.colab import files
files.download("risk_of_ruin_fixed_bet_pf12.csv")
Notas rápidas
Ruína: considero “zerou” quando o saldo ficou ≤ 0 em qualquer momento das 50 operações.

PF=12 com p=0,30 implica payoff médio do vencedor = 28R, assumindo perda média = 1R.

O resultado (bust_rate) depende fortemente do risk_%. Como o payoff vencedor é grande, ruína tende a cair bastante para riscos moderados, mas riscos altos elevam volatilidade dos resultados parciais — rode e veja os números.

Se quiser observar a sensibilidade:

troque P_WIN ou PROFIT_FACTOR;

ative USE_LOGNORMAL_WINS=True para variabilidade realista no ganho vencedor (média mantida em 28R);

mude N_TRADES ou N_SIMS.

Você disse:
como eu posso criar uma página com hospedagem gratuita pra rodar uma aplicação onde o usuário possa colocar todos esses inputs e ver esse rsultado?
O ChatGPT disse:

Pensou por 35s
Ótima ideia! Você tem duas rotas grátis e simples:

Python (recomendado): app em Streamlit hospedado no Streamlit Community Cloud (ou no Hugging Face Spaces).

100% front-end: página estática (HTML + JavaScript) no GitHub Pages (roda no navegador, sem servidor).

Como você já está em Python, vou te entregar a opção 1 completa (passo a passo + código). Se quiser a versão 100% front-end em JS depois, eu te mando também.

Opção 1 — Streamlit (Python) com hospedagem gratuita
Passo a passo de deploy (resumido)
Crie um repositório no GitHub com dois arquivos: app.py e requirements.txt (conteúdo abaixo).

Faça login no Streamlit Community Cloud e conecte seu GitHub.

Selecione o repositório, branch e o arquivo principal app.py e clique em Deploy.
(No Hugging Face Spaces é similar: “New Space” → template Streamlit → suba os arquivos.)

Vantagens: simples, aceita Python puro (NumPy/Pandas), tem UI rápida de montar.
Observação: free tiers podem hibernar após inatividade.

requirements.txt
streamlit==1.37.0
numpy==1.26.4
pandas==2.2.2
app.py (cole inteiro)
import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------
# Título e descrição
# -----------------------------
st.set_page_config(page_title="Risco de Ruína - Monte Carlo", layout="wide")
st.title("Risco de Ruína (Monte Carlo) — aposta fixa sobre o capital inicial")

st.markdown("""
Simula **10.000 cenários** com **50 operações** cada, sorteando vitórias e derrotas pela probabilidade de acerto.  
**Tamanho da aposta** = _percentual do capital inicial_ (fixo ao longo da simulação).  
A cada trade, o saldo é atualizado. **Conta como ruína** se em algum momento o saldo ficar **≤ 0**.

**Interpretação do PF (Profit Factor):**  
Assumimos perda média do trade perdedor = **1R**. Com **PF** e **p(acerto)** informados, o ganho médio do trade vencedor é calculado como  
**win_mean_R = PF × (1 - p) / p × 1R**.
""")

# -----------------------------
# Sidebar - Inputs
# -----------------------------
with st.sidebar:
    st.header("Parâmetros")

    start_equity = st.number_input("Capital inicial (R$)", min_value=0.0, value=10_000.0, step=100.0, format="%.2f")
    n_trades = st.number_input("Nº de operações por simulação", min_value=1, value=50, step=1)
    n_sims = st.number_input("Nº de simulações (Monte Carlo)", min_value=100, value=10_000, step=100)

    st.subheader("Probabilidade e PF")
    p_win = st.slider("Probabilidade de acerto (p)", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
    profit_factor = st.number_input("Profit Factor (PF)", min_value=0.0, value=12.0, step=0.5)

    st.subheader("Payoff do trade")
    loss_R = st.number_input("Perda média por trade perdedor (R)", min_value=0.0, value=1.0, step=0.1)
    use_lognormal = st.checkbox("Variabilidade no ganho do trade vencedor (lognormal)", value=False)
    lognormal_sigma = st.slider("Desvio padrão (log) do ganho vencedor", 0.1, 2.0, 0.6, 0.1, disabled=not use_lognormal)

    st.subheader("Níveis de risco (% do capital inicial por trade)")
    default_levels = "2, 5, 10, 15, 20, 25, 30"
    risk_levels_str = st.text_input("Lista separada por vírgulas", value=default_levels)
    seed = st.number_input("Seed (aleatoriedade reprodutível)", value=123, step=1)

    run_btn = st.button("Rodar simulação")

# -----------------------------
# Helpers
# -----------------------------
def parse_risk_levels(s: str):
    levels = []
    for part in s.split(","):
        part = part.strip().replace("%", "")
        if not part:
            continue
        try:
            levels.append(float(part)/100.0)
        except:
            pass
    return [r for r in levels if r > 0]

def sample_win_payoff_R(size, win_mean_R, use_lognormal, log_sigma, rng):
    if not use_lognormal:
        return np.full(size, win_mean_R, dtype=float)
    # lognormal com média = win_mean_R
    mu = np.log(win_mean_R) - 0.5 * (log_sigma**2)
    return rng.lognormal(mean=mu, sigma=log_sigma, size=size)

def simulate_busts_for_risk(risk_pct, n_sims, n_trades, start_equity, p_win, win_mean_R, loss_R, use_lognormal, log_sigma, base_seed):
    """Retorna (bust_count, bust_rate, exemplo_de_saldo) para um nível de risco."""
    rng = np.random.default_rng(base_seed)
    busts = 0
    bet_amount = risk_pct * start_equity  # aposta fixa em R$ por trade

    # Vamos salvar 1 amostra de trajetória para mostrar
    sample_equity_path = None
    captured = False

    for s in range(n_sims):
        equity = start_equity
        outcomes = rng.random(n_trades) < p_win  # True = win
        win_Rs = sample_win_payoff_R(n_trades, win_mean_R, use_lognormal, log_sigma, rng)

        busted = False
        path = [equity]
        for i, win in enumerate(outcomes):
            if win:
                equity += bet_amount * win_Rs[i]
            else:
                equity -= bet_amount * loss_R

            path.append(equity)
            if equity <= 0:
                busted = True
                break

        if busted:
            busts += 1

        if not captured:
            sample_equity_path = path
            captured = True

    return busts, 100.0*busts/n_sims, sample_equity_path

# -----------------------------
# Run
# -----------------------------
if run_btn:
    if p_win <= 0 and profit_factor > 0:
        st.error("p(acerto) precisa ser > 0 para definir o ganho médio vencedor.")
        st.stop()

    risk_levels = parse_risk_levels(risk_levels_str)
    if not risk_levels:
        st.error("Informe pelo menos um nível de risco válido (ex.: 2, 5, 10).")
        st.stop()

    # Ganho médio do trade vencedor (em R), consistente com PF
    # PF = (p * W_mean) / ((1 - p) * L_mean)  =>  W_mean = PF * (1-p)/p * L_mean
    if p_win == 0:
        win_mean_R = 0.0
    else:
        win_mean_R = profit_factor * (1.0 - p_win) / max(p_win, 1e-12) * loss_R

    results = []
    example_paths = {}

    with st.spinner("Simulando..."):
        for r in risk_levels:
            bust_count, bust_rate, path = simulate_busts_for_risk(
                risk_pct=r,
                n_sims=n_sims,
                n_trades=n_trades,
                start_equity=start_equity,
                p_win=p_win,
                win_mean_R=win_mean_R,
                loss_R=loss_R,
                use_lognormal=use_lognormal,
                log_sigma=lognormal_sigma,
                base_seed=seed + int(r*10_000),
            )
            results.append({
                "risk_%": int(round(r*100)),
                "bust_count": bust_count,
                "bust_rate_%": bust_rate
            })
            example_paths[int(round(r*100))] = path

    df = pd.DataFrame(results).set_index("risk_%").sort_index()
    st.subheader("Resultados")
    st.dataframe(df.style.format({"bust_rate_%": "{:.2f}"}), use_container_width=True)

    st.download_button(
        label="Baixar CSV",
        data=df.to_csv(index=True).encode("utf-8"),
        file_name="risk_of_ruin_fixed_bet.csv",
        mime="text/csv"
    )

    # Plot opcional: taxa de ruína por risco
    st.subheader("Taxa de ruína por nível de risco")
    st.bar_chart(df["bust_rate_%"])

    # Exibir 1 trajetória de exemplo por risco (opcional)
    with st.expander("Ver amostras de trajetórias de saldo (1 por risco)"):
        import altair as alt
        for rk, path in example_paths.items():
            st.markdown(f"**Risco {rk}%** — amostra de trajetória")
            dt = pd.DataFrame({"trade": list(range(len(path))), "equity": path})
            chart = alt.Chart(dt).mark_line().encode(x="trade:Q", y="equity:Q").properties(height=180)
            st.altair_chart(chart, use_container_width=True)

else:
    st.info("Defina os parâmetros na barra lateral e clique em **Rodar simulação**.")
Quer uma página 100% estática (sem Python)?
Também dá: eu crio um HTML + JavaScript puro com toda a simulação rodando no navegador. Você hospeda no GitHub Pages (gratuito) em poucos cliques. Vantagem: nunca hiberna e não depende de servidor Python. Se topar, eu te mando o pacote index.html pronto.

Você disse:
como adicionar os arquivos no repositório github?
O ChatGPT disse:

Pensou por 27s
Segue um passo a passo rápido (três jeitos). Use o que preferir.

Opção A — Pelo navegador (o mais simples)
Entre no GitHub e clique em New (novo repositório).

Dê um nome (ex.: risco-de-ruina-streamlit) e clique em Create repository.

Na página do repositório, clique em Add file → Upload files.

Arraste e solte seus arquivos (ex.: app.py, requirements.txt) para a tela.

Em Commit changes, escreva uma mensagem (ex.: “primeiro commit”) e clique em Commit.

Pronto: seus arquivos estão no GitHub.

Opção B — Via Git no terminal (Windows/Mac/Linux)
Requer ter o Git instalado.

# 1) Crie uma pasta local e entre nela
mkdir risco-de-ruina-streamlit
cd risco-de-ruina-streamlit

# 2) Coloque aqui seus arquivos (app.py, requirements.txt, etc.)

# 3) Inicie o repositório local e faça o primeiro commit
git init
git add .
git commit -m "primeiro commit"

# 4) Crie um repositório VAZIO no GitHub (web) e copie a URL "origin"
#    Exemplo de URL: https://github.com/SEU_USUARIO/risco-de-ruina-streamlit.git

git branch -M main
git remote add origin https://github.com/SEU_USUARIO/risco-de-ruina-streamlit.git
git push -u origin main
Dica: quando o Git pedir usuário/senha, hoje o GitHub usa Personal Access Token no lugar de senha.

Opção C — GitHub Desktop (sem terminal)
Instale GitHub Desktop e faça login.

File → New repository… para criar localmente.

Coloque seus arquivos na pasta do repositório.

Em Changes, escreva a mensagem e clique Commit to main.

Clique Publish repository para subir para o GitHub.

O que subir para este projeto (mínimo)
app.py (o código do Streamlit que te passei)

requirements.txt

(opcional) .gitignore com:

__pycache__/
*.pyc
.env
Depois de subir, você já consegue fazer o deploy gratuito no Streamlit Community Cloud: conecte sua conta, escolha o repositório, selecione app.py como arquivo principal e publique.

Se quiser, eu reviso seu repositório (nome/arquivos/estrutura) e já deixo tudo pronto para o deploy.




Nenhum arquivo escolhidoNenhum arquivo escolhido
O ChatGPT pode cometer erros. Considere verificar informações importantes.
