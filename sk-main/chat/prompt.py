import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.kernel import SKFunctionBase


class PromptManager:
    
    def __init__(self) -> None:
        self.kernel = sk.Kernel()
        deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
        self.kernel.add_chat_service(
            "development",
            AzureChatCompletion(deployment_name=deployment, base_url=endpoint, api_key=api_key)
        )

    def prompt(self, instructions: str) -> SKFunctionBase:
        return self.kernel.create_semantic_function("""{{$input}} \n\n %s""" % instructions)

    def run(self, instructions: str, content: str) -> None:
        return self.prompt(instructions)(content)



if __name__ == "__main__":
    manager = PromptManager()
    print(manager.run("write in one single line", "What is the difference between 2 apples and 2 bananas?"))