from __future__ import annotations

import uuid
import logging
from typing import Type, Optional

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.kernel import KernelFunction

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient

from app.agents.agents import MemoryAgent
from app.schemas.agents import ChatSchema, SearchEngineSchema
from app.tools.memories import CosmosMongoMemory
from app.utils.tracker import evaluate_performance


logger: logging.Logger = logging.getLogger(__name__)


class SimpleRAG(MemoryAgent):

    def _config_service(
        self,
        chat_name: str,
        completion: Type[AzureChatCompletion] = AzureChatCompletion,
        schema: ChatSchema = ChatSchema(),
        memory: Optional[CosmosMongoMemory] = None
    ) -> None:
        """
        Configures and adds a chat service to the kernel.

        Args:
            chat_name (str): Name of the chat service to configure.
            *args: Variable length argument list for the chat service.
            **kwargs: Arbitrary keyword arguments for the chat service.
        """
        self.kernel.add_chat_service(
            chat_name,
            completion(**schema.model_dump())
        )
        if memory:
            self._chat_history(memory)

    @evaluate_performance
    async def prompt(self, prompt: str, **kwargs) -> KernelFunction:
        """
        Creates and returns a semantic function based on the given prompt and tool mappings.

        Args:
            prompt (str): The prompt to use for creating the semantic function.
            **kwargs: Arbitrary keyword arguments for the semantic function.

        Returns:
            KernelFunctionBase: The created semantic function.
        """

        prompt_template = """
        You are a research assistant.\n
        You will write a summary of the research, with a brief introduction and a review of the topic.\n
        Your answer should be structured in topics, based on the content of the chat history and the presaved terms of the research.\n
        Your answer should have at least 1000 words.\n
        \n------------------------------\n
        Consider the following chat history:\n
        {{$chat_history}}
        \n------------------------------\n
        Consider the following presaved researched documents:\n
        {{$RESEARCH_TOPICS}}
        \n------------------------------\n
        Provide a summary to a research based on the following question:\n
        {{$input}}
        """
    
        self.context['chat_history'] = await self.kernel.memory.search('ragMemory', prompt, 10)
        self.context['RESEARCH_TOPICS'] = await self.augmented_retrieve(prompt)
        self.context['input'] = prompt
        return self.kernel.create_semantic_function(prompt_template, **kwargs)

    async def augmented_retrieve(self, prompt: str) -> str:
        """
        Performs an augmented retrieval of research documents based on the provided prompt.

        Args:
            prompt (str): The prompt to use for the retrieval.

        Returns:
            str: Aggregated content of the top relevant research documents.
        """
        schema = SearchEngineSchema()
        search_engine = SearchClient(
            endpoint=schema.endpoint,
            index_name=schema.index_name,
            credential=AzureKeyCredential(schema.search_key)
        )
        research_docs = ''
        async with search_engine:
            results = await search_engine.search(search_text=prompt, top=10)
            async for result in results:
                if result.get('@search.score', 0) > 20:
                    research_docs += result['content'] + '\n'
        return research_docs


class OneShotRAG(MemoryAgent):

    def _config_service(
        self,
        chat_name: str,
        completion: Type[AzureChatCompletion] = AzureChatCompletion,
        schema: ChatSchema = ChatSchema(),
        memory: Optional[CosmosMongoMemory] = None
    ) -> None:
        """
        Configures and adds a chat service to the kernel.

        Args:
            chat_name (str): Name of the chat service to configure.
            *args: Variable length argument list for the chat service.
            **kwargs: Arbitrary keyword arguments for the chat service.
        """
        self.kernel.add_chat_service(
            chat_name,
            completion(**schema.model_dump())
        )
        if memory:
            self._chat_history(memory)

    @evaluate_performance
    async def prompt(self, prompt: str, **kwargs) -> KernelFunction:
        """
        Creates and returns a semantic function based on the given prompt and tool mappings.

        Args:
            prompt (str): The prompt to use for creating the semantic function.
            **kwargs: Arbitrary keyword arguments for the semantic function.

        Returns:
            KernelFunctionBase: The created semantic function.
        """

        prompt_template = """
        You are a research assistant.\n
        You will write a summary of the research, with a brief introduction and a review of the topic.\n
        Your answer should be structured in topics, based on the content of the chat history and the presaved terms of the research.\n
        Your answer should have at least 1000 words.\n
        \n------------------------------\n
        Use the following example to improve your answer:\n
        QUESTION:
        Perform an analysis of the transformer architecture
        ANSWER:
        # Introduction to Transformer Architecture
        In recent years, the field of natural language processing (NLP) has been revolutionized by the introduction of the Transformer architecture. This breakthrough was first introduced in the seminal paper "Attention Is All You Need" by Vaswani et al. in 2017. The Transformer model eschews the previously dominant sequence-to-sequence architectures that relied heavily on recurrent neural networks (RNNs) and convolutional neural networks (CNNs), and instead uses a self-attention mechanism to process sequential data.
        The shift to Transformers has led to the development of various state-of-the-art models that have set new standards in a wide array of NLP tasks, including but not limited to language understanding, translation, question-answering, and summarization. This summary will delve into the core concepts of the Transformer architecture, its advantages, applications, and the subsequent developments it has spurred in the field of artificial intelligence.
        # Core Concepts of Transformer Architecture
        ## Self-Attention Mechanism
        The linchpin of the Transformer architecture is the self-attention mechanism. This allows the model to weigh the significance of each part of the input data differently, enabling it to capture context more effectively. Self-attention computes a score for each word in a sentence in relation to every other word, which determines how much focus should be placed on other parts of the input when encoding a particular word.
        ## Positional Encoding
        Since Transformers do not inherently process sequential data as RNNs do, they require positional encodings to maintain the order of words. Positional encodings are added to the input embeddings to provide the model with information about the position of the words in the sequence.
        ## Multi-Head Attention
        Transformers utilize multi-head attention to extend the self-attention mechanism across multiple 'heads', allowing the model to capture different types of relationships in the data across different representation subspaces at different positions.
        ## Encoder-Decoder Structure
        The original Transformer model is composed of an encoder to process the input and a decoder to generate the output. Each consists of a stack of identical layers that contain multi-head self-attention and feed-forward neural network components.
        # Advantages of Transformer Architecture
        ## Parallelization
        Unlike RNNs, which process data sequentially, Transformers can handle different parts of the sequence simultaneously, which makes them highly parallelizable and significantly faster in training.
        ## Scalability
        The Transformer's ability to parallelize processing also allows it to scale effectively with the addition of more data and compute resources, which has led to the creation of massive models like GPT and T5 that have billions of parameters.
        ## Long-Range Dependencies
        The self-attention mechanism can theoretically capture relationships between words regardless of their distance in the sequence, which helps in understanding the context and nuances of the language better than RNNs or CNNs.
        # Applications of Transformer Architecture
        ## Machine Translation
        Transformers have been employed to create models that provide translations that are often indistinguishable from human translations, handling complex languages and idiomatic expressions effectively.
        ## Text Summarization
        Models based on Transformers can produce coherent and concise summaries of long documents, which is useful in digesting large amounts of information quickly.
        ## Question Answering
        Transformers have been used to develop systems that can understand and answer questions with high accuracy, which is essential for search engines and virtual assistants.
        ## Sentiment Analysis
        These models can understand the sentiment behind texts, making them valuable tools for social media monitoring and market research.
        # Subsequent Developments
        ## BERT
        BERT (Bidirectional Encoder Representations from Transformers) represents a significant leap forward by pre-training on a large corpus of text and then fine-tuning on specific tasks. Its bidirectional nature allows it to understand the context of a word based on all of its surroundings.
        ## GPT Models
        Generative Pretrained Transformer (GPT) models take the Transformer architecture and apply it in a generative manner, allowing for the creation of text that can be remarkably coherent and contextually relevant.
        ## T5 and Other Variants
        T5, or Text-to-Text Transfer Transformer, takes the concept further by converting every NLP problem into a text-to-text format, enabling a more unified approach to NLP tasks.
        # Conclusion
        The Transformer architecture has undeniably altered the landscape of NLP and continues to be the backbone of the most advanced models in the field. Its core concepts of self-attention and the encoder-decoder framework have paved the way for more efficient, accurate, and context-aware models. The subsequent developments, including BERT and GPT, have showcased the architecture's versatility and power, making it a cornerstone of modern NLP research and applications. As the field advances, we can expect the Transformer to evolve further, driving the next generation of AI breakthroughs.
        "
        \n------------------------------\n
        Consider the following presaved researched documents:\n
        {{$RESEARCH_TOPICS}}
        \n------------------------------\n
        Provide a summary to a research based on the following question:\n
        {{$input}}
        """
    
        self.context['chat_history'] = await self.kernel.memory.search('ragMemory', prompt, 10)
        self.context['RESEARCH_TOPICS'] = await self.augmented_retrieve(prompt)
        self.context['input'] = prompt
        return self.kernel.create_semantic_function(prompt_template, **kwargs)

    async def augmented_retrieve(self, prompt: str) -> str:
        """
        Performs an augmented retrieval of research documents based on the provided prompt.

        Args:
            prompt (str): The prompt to use for the retrieval.

        Returns:
            str: Aggregated content of the top relevant research documents.
        """
        schema = SearchEngineSchema()
        search_engine = SearchClient(
            endpoint=schema.endpoint,
            index_name=schema.index_name,
            credential=AzureKeyCredential(schema.search_key)
        )
        research_docs = ''
        async with search_engine:
            results = await search_engine.search(search_text=prompt, top=10)
            async for result in results:
                if result.get('@search.score', 0) > 20:
                    research_docs += result['content'] + '\n'
        return research_docs
