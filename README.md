# CoolSync+

## Predictive Cooling Control for AI Data Centers using LSTM + DQN ❄️🌍🤖

CoolSync+ is an intelligent predictive cooling optimization system designed for modern AI data centers handling large-scale LLM workloads such as ChatGPT, Gemini, Copilot, and enterprise inference systems.

As AI workloads increase, GPU servers generate sudden thermal bursts that traditional reactive cooling systems fail to handle efficiently. Conventional PID controllers respond only after temperature rises, while cooled air from CRAC systems takes nearly 3 minutes to reach server racks, creating overheating risks, energy waste, and poor Power Usage Effectiveness (PUE).

CoolSync+ solves this problem using Machine Learning + Reinforcement Learning by predicting thermal demand before overheating happens and proactively optimizing cooling decisions.

---

# Project Objective

Build a predictive cooling control system that:

* predicts GPU heat spikes before they happen
* reduces cooling energy consumption
* eliminates ASHRAE thermal violations
* improves Power Usage Effectiveness (PUE)
* lowers carbon emissions
* supports sustainable AI infrastructure

---

# System Architecture

CoolSync+ uses a 4-stage intelligent pipeline:

```text
User Prompt
   ↓
Stage 1 — Prompt Classification (GBT)
   ↓
Stage 2 — GPU Heat Mapping
   ↓
Stage 3 — LSTM Temperature Prediction
   ↓
Stage 4 — DQN Cooling Controller
   ↓
Proactive Cooling Decision
```

---

# Core Technologies

## Machine Learning

* Gradient Boosting Tree (GBT)
* LSTM Temperature Forecasting

## Reinforcement Learning

* Markov Decision Process (MDP)
* Bellman Equation
* Q-Learning
* Deep Q-Network (DQN)
* Experience Replay
* Target Networks
* Double DQN

## Thermodynamics + HVAC

* FOPDT (First Order Plus Dead Time)
* PID vs Predictive Cooling
* ASHRAE Thermal Compliance
* COP-based Chiller Optimization
* Fan Cube Law

---

# Key Results

Compared to traditional PID cooling:

## Cooling Energy Savings

* 42–50% cooling energy reduction

### Workload Examples

* Short Query → 49.6% savings
* Very Long Query → 42.5% savings

## Peak Hour Scenario

* 50.6% energy savings
* Zero ASHRAE violations

## PUE Improvement

```text
1.56 → 1.476
```

## Environmental Impact

* lower electricity consumption
* lower CO₂ emissions
* less water waste
* reduced hardware thermal stress
* lower e-waste
* greener AI infrastructure

---

# Why This Matters for Canada 🇨🇦

Canada is becoming a major AI + data center hub across:

* Toronto
* Montreal
* Waterloo
* Vancouver
* Calgary

More AI means more cooling demand.

CoolSync+ helps reduce:

* summer peak electricity pressure
* carbon emissions
* operational costs
* sustainability risks

especially in Ontario and Alberta where grid demand becomes expensive during peak periods.

---

# Repository Structure

```text
CoolSync_CSCN8020/
│
├── README.md
├── CoolSync_Technical_Report_EN.md
│
├── backend/
├── greeninference/
├── models/
├── data/
├── scenarios/
├── results/
│
└── requirements.txt
```

---

# Full Technical Report

For full technical details including:

* equations
* LSTM architecture
* DQN reward function
* Bellman equation
* FOPDT modeling
* PID vs DQN comparison
* PUE calculations
* datasets
* validation methodology
* references

please see:

## CoolSync_Technical_Report_EN.md

---

# Final Message

## The future of AI depends on how we cool it.

CoolSync+ proves that Reinforcement Learning can solve real industrial sustainability problems far beyond robotics and gaming.

It is not just smarter AI.

It is smarter infrastructure.
