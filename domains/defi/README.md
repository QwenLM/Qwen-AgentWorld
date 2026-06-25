# GenTech World — DeFi Domain

**Purpose:** Add DeFi capabilities to Qwen-AgentWorld. The 8th domain: trading, yield, risk, payments.

## What This Adds

| Domain | What Agents Learn |
|--------|-------------------|
| **Trading** | Buy/sell decisions, market timing, position sizing |
| **Yield** | LP management, rebalancing, fee optimization |
| **Risk** | Rug detection, impermanent loss, smart contract risk |
| **Payments** | x402 micropayments, cross-chain settlements |

## Training Data Sources

### From GenTech Labs
- LP position history (AVAX/USDC, Curve, Bid-Ask)
- Trade decisions and reasoning
- Market observations from cron jobs
- API usage patterns from x402 logs
- Risk scoring data from Rugcheck

### From Agent Arena
- Harvest mode decisions
- Dry Powder mode triggers
- Bullish mode deployments
- Payday mode withdrawals

### From Market Data
- Price feeds (Pyth, DexScreener)
- Gas prices (multiple chains)
- Protocol TVL (DefiLlama)
- Risk scores (Rugcheck)

## AgentWorldBench for DeFi

### Metrics
- **Trade Success Rate** — % of profitable trades
- **Yield Optimization** — APY vs benchmark
- **Risk Avoidance** — % of rug attempts caught
- **Payment Efficiency** — x402 success rate

### Benchmarks
- **DeFi Trading** — Buy/sell decisions on 10 pairs
- **LP Management** — Rebalancing across 3 DEXs
- **Risk Assessment** — Evaluate 20 tokens
- **Cross-Chain Settlement** — Pay across 3 chains

## Integration with Agent Arena

### Pre-Trade Simulation
1. Agent receives trade signal
2. AgentWorld simulates outcome
3. Agent decides: execute or skip
4. Agent executes with confidence

### Yield Optimization
1. Agent analyzes current position
2. AgentWorld simulates 3 scenarios
3. Agent picks best scenario
4. Agent rebalances accordingly

### Risk Management
1. Agent receives token alert
2. AgentWorld evaluates risk factors
3. Agent decides: safe, warning, or rug
4. Agent acts accordingly

## Files

- `README.md` — This file
- `trajectories/` — Training data
- `benchmarks/` — DeFi AgentWorldBench
- `prompts/` — DeFi-specific prompts
- `eval/` — DeFi evaluation metrics

---

*The 8th domain. DeFi meets world models.*
