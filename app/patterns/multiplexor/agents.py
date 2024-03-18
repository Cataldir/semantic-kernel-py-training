
from semantic_kernel import Kernel
from app.agents import MemoryAgent


def multiplexor_selection(prompt: str, kernel: Kernel, **kwargs):
    prompt = prompt.lower()
    prompt_template = """
        You are a multiplexor that selects which agent is going to answer the question.\n
        You will work with four different contexts, as below:\n
        \n------------------------------\n
        BUSINESS CONTEXT
        Context: If the question has a business context, you should select the agent that has the best knowledge about the business.\n
        \n------------------------------\n
        Risks and Mitigators CONTEXT
        Context: If the question asks about a know risk with proper mitigator, you should select the agent that has the best knowledge about the risk and mitigator.\n
        \n------------------------------\n
        In your answer, you will take the User`s prompt and create a Python Dictionary that is composed of the following:\n
        [
            {
                'agent': 'agent_name',
                'agent_type: 'BusinessAgent',
                'prompt': USER Prompt contextualized to the agent speciality,
            },
            {
                'agent': 'agent_name',
                'agent_type: 'RisksAgent',
                'prompt': USER Prompt contextualized to the agent speciality,
            }
        ]
        \n------------------------------\n
        Your answer should contain only the list of dictionaries that you created.
        \n------------------------------\n
        This is the user question:\n
        {{$request}}
        """
    context = kernel.create_new_context()
    context['input'] = prompt
    return kernel.create_semantic_function(prompt_template, **kwargs)


class BusinessAgent(MemoryAgent):

    def plan(self, prompt: str):
        search_results: str = self.search_memory(prompt)
        self.context['search_result'] = search_results
        return self.evaluate_business_risks(prompt)

    def evaluate_business_risks(self, prompt: str):
        prompt_template = """
        You are a business risk analyst.\n
        You will work with four different contexts, as below:\n
        \n------------------------------\n
        Consider the following search context:
        {{$search_result}}
        \n------------------------------\n
        BUSINESS CONTEXT
        Context: If the question has a business context, you should select the agent that has the best knowledge about the business.\n
        \n------------------------------\n
        Risks and Mitigators CONTEXT
        Context: If the question asks about a know risk with proper mitigator, you should select the agent that has the best knowledge about the risk and mitigator.\n
        \n------------------------------\n
        In your answer, you will take the User`s prompt and create a Python Dictionary that is composed of the following:\n
        [{
            'agent': 'agent_name',
            'agent_type: 'BusinessAgent',
            'prompt': USER Prompt contextualized to the agent speciality,
        },
        {
            'agent': 'agent_name',
            'agent_type: 'RisksAgent',
            'prompt': USER Prompt contextualized to the agent speciality,
        }]
        \n------------------------------\n
        Your answer should contain only the list of dictionaries that you created.
        \n------------------------------\n
        This is the user question:\n
        {{$request}}
        """
        context = kernel.create_new_context()
        context['input'] = prompt
        return kernel.create_semantic_function(prompt_template, **kwargs)