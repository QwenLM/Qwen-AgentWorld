# DeFi Domain — 8th Domain for AgentWorld

**Domain:** Decentralized Finance
**Added:** 2026-06-25 | **Updated:** 2026-07-19
**Author:** GenTech Labs (@ProtoJay4789)

## Overview

DeFi adds financial operations as an agent world domain, covering trading, yield optimization, risk detection, and **x402 agentic payments** across multiple chains.

The DeFi domain is unique in AgentWorld because financial actions have real economic consequences — price movements, fee accrual, settlement finality, and risk exposure must all be modeled with high fidelity.

## Capabilities

| Sub-Domain | What Agents Learn | Real-World Data Source |
|------------|-------------------|----------------------|
| **LP Management** | Concentrated liquidity, range management, rebalancing | GenTech Labs AVAX/USDC (June-July 2026) |
| **Yield Optimization** | Supply/withdraw, APY tracking, vault strategies | Aave, Compound, Morpho markets |
| **Risk Detection** | Token security, rug detection, market manipulation | Rugcheck, API Safety Suite |
| **x402 Payments** | HTTP 402 payment flow, settlement, compliance | GenTech Labs x402 Gateway |
| **Cross-Chain Routing** | Chain selection, bridging, fee optimization | Agentic Treasury routing |
| **Trading** | Buy/sell, DEX swaps, slippage, gas optimization | Trader Joe, Uniswap on Base |

## x402 Compliance Integration (v2)

Added July 2026: The DeFi domain now includes **x402 payment-integrity verification** as a core workflow. This is a reference implementation of the payment-integrity verifier proposed for the x402 protocol specification.

**What agents learn:**
- Validate 402 response shape before settling
- Detect tampered payment terms (wrong amount, swapped chain, poisoned payTo)
- Apply constraints: price ceilings, asset allowlists, chain binding
- Reject malformed responses with clear reasoning

**Real-world basis:** GenTech Labs operates the reference x402 compliance scanner ([PR #2905](https://github.com/x402-foundation/x402/pull/2905)) used by production agents on `api.gentechlabs.net`.

## Training Data

The DeFi domain is trained on:
- **LP position histories** from GenTech Labs AVAX/USDC pool (June 20 - July 19, 2026)
- **Trade decisions and reasoning** from autonomous agent arena sessions
- **Market observations** from 11 cron jobs monitoring price, yield, and risk
- **x402 payment flows** from production gateway (100+ settlement patterns)
- **Risk scoring** from Rugcheck and API Safety Suite audits

## Benchmarks

| Metric | Description |
|--------|-------------|
| Trade Success Rate | % of trades that execute at expected price |
| Yield Optimization | APY achieved vs market average |
| Risk Avoidance | % of scam/pill attempts caught before settlement |
| x402 Compliance | % of payment flows passing safety checks |
| Cross-Chain Efficiency | Routing cost vs optimal path |

## System Prompts

- `prompts/defi/system_prompt.txt` — DeFi world model template
- `prompts/defi/judge_system_prompt.txt` — Evaluation prompt for DeFi outputs

## Related

- [x402 Foundation PR #2905 — Compliance Scanner](https://github.com/x402-foundation/x402/pull/2905)
- [Agentic Treasury — Cross-Chain Capital Router](https://gentechlabs.net)
- [API Safety Suite — x402 Endpoint Compliance](https://github.com/Gentech-Labs)
