# Hawkes in finance
This project analyze the microstructure of the Bitcoin (BTC-USD) market by modeling volatil-
ity clustering using a temporal Hawkes process. Using hourly log-returns, we isolate extreme
price movements and retrive the estimation of the parameters via Maximum Likelihood Estimation
(MLE), using gradient-based optimization. The empirical results quantify the reflexive nature of
the crypto market, revealing a branching ratio (𝛼) of roughly 0.53; which indicate that over half
of severe volatility shocks are endogenously driven by internal contagion rather than exogenous
news. Furthermore, the estimated decay parameter (𝛽) indicate a relaxation half-life of approxi-
mately 14.5 hours. Beyond the theoretical framework, we make use of the continuous conditional
intensity 𝜆(𝑡) as quantitative signal. In a trading environment, this signal serves as risk proxy,
allowing liquidity providers to defensively scale exposures or asymmetrically widen bid-ask spreads
to monetize during periods of extreme market hysteresis.
