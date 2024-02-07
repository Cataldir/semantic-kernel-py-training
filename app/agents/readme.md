# Agents on Semantic Kernel

Agents are enabled in SK [through experiments](https://github.com/microsoft/semantic-kernel/tree/main/dotnet/src/Experimental/Agents), but you can also create them depending on your need. They are, as said before, a logically-oriented entity that enables interaction with the LLM through a series of predefined actions. The agent itself is an abstraction of the process one should take in order to solve a specific structure of the problem which the LLM is going to be used on.

## Structuring Agents

In this repository, we have build a Agent class by applying the [template method](https://refactoring.guru/design-patterns/template-method). The __call__ dunder method is used as the template method for the class, and the structure of the agent is meant to use Semantik Kernell context to improve the capacity and drive the behaviour according to a set of methods, closely related to the implementation of plans. The idea is to set the methods that you want the agent to perform within its context, by providing a Python dictionary with the name of the pattern that should be under the context and assigning corresponding callable functions or coroutine objects. This dictionary acts as a dynamic planner, allowing the Agent to adapt and respond to various scenarios using the semantic kernel's rich contextual understanding. Each method linked in the dictionary can leverage the semantic kernel's features, like natural language processing and contextual analysis, to execute complex tasks and make informed decisions. The Agent class, being abstract, serves as a blueprint for creating more specialized agents, each tailored to specific applications while sharing a common, robust foundation of semantic understanding and adaptability.

## Expanding the "Agent" Class

Expanding the "Agent" class effectively requires a nuanced understanding of both the foundational aspects of the class and the specific requirements of the context in which the agent will operate. The goal is to enhance the agent's functionality while maintaining the integrity and flexibility of its core structure.

### Embracing Specialization Through Subclassing

The creation of specialized subclasses, like the Researcher class, exemplifies a strategic approach to expansion. This method preserves the original structure of the Agent class while introducing new functionalities tailored to specific needs. Subclassing allows us to leverage the robustness of the base class and extend its capabilities in a controlled and organized manner. This ensures that the core functionalities remain intact and reusable across different contexts.

### Integration with External Services

Integrating external services such as Azure AI or Semantic Kernel into the agent is crucial for enhancing its capabilities. This integration should not be arbitrary but thoughtfully aligned with the agent's intended functionalities. For instance, incorporating Azure's AI services for advanced chat functionalities can significantly amplify the agent’s ability to process and understand complex language structures, thereby making it more effective in its operations.

### Leveraging Asynchronous Programming

The use of asynchronous programming is not just a technical choice, but a strategic one. In an environment where agents are expected to perform multiple tasks efficiently, embracing async-await patterns in methods, like _chat_history, enables the agent to handle IO-bound tasks more effectively. This approach ensures that the agent remains responsive and agile, especially in scenarios involving database access or network communication.

### Dynamic Configuration and Argument Flexibility

The ability to dynamically configure services and accept a wide range of arguments reflects an adaptive design philosophy. Methods designed to configure different aspects of the agent, such as the _config_service method, demonstrate an understanding of the varied and evolving nature of the tasks an agent may encounter. The use of variable length argument lists and arbitrary keyword arguments in these methods enhances the agent's adaptability, allowing it to operate effectively in diverse scenarios.

### Importance of Logging and Structured Data Handling

Incorporating logging and structured data handling is not just a best practice but a necessity for maintaining and monitoring the agent’s health and performance. Logging provides visibility into the agent's operations, aiding in debugging and performance optimization. Similarly, the use of schemas for data validation and configuration management through settings ensures that the agent processes data consistently and operates reliably in different environments.