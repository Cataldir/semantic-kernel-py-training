# Semantic Kernel (Python) - Training and Understanding

The beauty of Semantic Kernel relies on the elegant way that it orchestrate the communication with the LLMs, by providing concise and well-defined separations between semantic functions (those that expect an interaction with the LLM) from native functions (those that doesn't). Also, in a different strategy from LangChain, Semantic Kernel specializes in the communication with the LLM through a kernel of actions, which is more straighforward and in accordance with microservices architectural patterns.

## About this repository

This is a repository majorly oriented to understand the capabilities and limitations of Semantic Kernel as a Python package.
This repository is not meant to be used as a reference for extension, serving the only purpose of understanding the package semantics and overall strategies for using it.

## Learning Objectives

- Understand how to apply the concept of "Agents" with semantic kernel
- Understand what a "Plan" is in the context of Semantic Kernel
- Understand how to prepare Tools and plugins with Semantic Kernel

## Agents

Agents are entities that control the flow of action and interaction with the user. They are empowered by the LLM (which is the "brain") and a set of tools and
procedures that should be timelly called whenever you have the need in the logic flow of response.

From the perspective of Semantic Kernel orchestration, an AI agent is a modular abstraction that can possess a persona, can perform actions in response to user input, and can easily communicate with other agents. You might also view an agent from an AI-as-a-service perspective or as an autonomous worker.

[Check this explanation on agents for SK](https://github.com/Cataldir/semantic-kernel-py-training/tree/main/app/agents)
[And this is the official way of doing it](https://learn.microsoft.com/en-us/semantic-kernel/agents/)

## Context



## Tooling and Plugins

