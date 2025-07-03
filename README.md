# Multi-Agent Search Planner

This implements the logic that powers intelligent agent behavior within the Multi-Agent Search simulation. It provides a framework for modeling search problems and executing a variety of classical search strategies in a dynamic environment.

The system is designed to simulate real-time decision-making, where multiple agents navigate a shared space, pursue goals and interact with their surroundings. The logic supports both single-agent and multi-agent settings, enabling agents to reason individually or competitively.

At its core, the framework defines abstract representations of search problems, including state spaces, goal conditions, successor generation and cost models. Agents use these definitions to plan actions using algorithms such as minimax, expectimax and reflex-based strategies.

The simulation engine manages game state, agent movement and visualization, ensuring that the decision-making logic is executed within an interactive environment. The modular design allows for experimentation with different agent behaviors, evaluation functions and planning strategies.
