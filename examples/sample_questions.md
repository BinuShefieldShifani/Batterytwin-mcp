# Example questions

These show the kinds of questions the agent answers by calling the twin's MCP
tools. Each maps to one or more physics simulations under the hood.

## Performance & runtime
- "What cell are we modelling, and what is its voltage window?"
- "How long can the cell run when discharged at 2C?"
- "Compare deliverable capacity at 0.5C versus 2C."
- "Does discharging at 0 C reduce how much capacity I get out?"

## Charging
- "How long does a full CC-CV charge take at 0.5C?"
- "Which is fastest to charge: 0.3C, 0.5C, or 1C - and what do I trade off?"
- "If I need to charge in under 100 minutes, what C-rate should I use?"

## Ageing / cycle life
- "How much capacity is lost over 10 cycles at 40 C?"
- "Is fast charging at 1C worse for cycle life than 0.5C over 15 cycles?"
- "What is the dominant degradation mechanism in this cell?"

## Multi-step reasoning (the agent chains tools)
- "I want the fastest charge that still keeps fade under a threshold over 10
  cycles - what do you recommend?"
- "Summarise this cell's behaviour across discharge, charging, and ageing for a
  spec sheet."
