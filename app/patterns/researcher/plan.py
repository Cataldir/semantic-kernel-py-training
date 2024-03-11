# Copyright (c) Microsoft. All rights reserved.

import os
from typing import TYPE_CHECKING
# The `Plan` class is likely used for representing a plan or
# a sequence of actions to achieve a specific goal in the
# context of planning algorithms. It may contain information
# about the steps, actions, or tasks that need to be
# executed in a specific order to reach a desired outcome.
# This class could be used in conjunction with other
# planning classes like `BasicPlanner`, `SequentialPlanner`,
# `StepwisePlanner`, and `ActionPlanner` to help organize
# and execute a series of actions towards a goal.

from semantic_kernel.kernel import Kernel
from semantic_kernel.planners.plan import Plan
from semantic_kernel.planners.sequential_planner.sequential_planner_config import (
    SequentialPlannerConfig,
)
from semantic_kernel.planners.sequential_planner.sequential_planner_parser import (
    SequentialPlanParser,
)
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig


SEQUENTIAL_PLANNER_DEFAULT_DESCRIPTION = (
    "Given a request or command or goal generate a step by step plan to "
    + "fulfill the request using functions. This ability is also known as decision making and function flow"
)

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PROMPT_CONFIG_FILE_PATH = os.path.join(CUR_DIR, "Plugins/SequentialPlanning/config.json")
PROMPT_TEMPLATE_FILE_PATH = os.path.join(CUR_DIR, "Plugins/SequentialPlanning/skprompt.txt")


def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


class ResearchPlanner:
    RESTRICTED_PLUGIN_NAME = "ResearcherPlanner_Excluded"

    config: SequentialPlannerConfig
    _context: "KernelContext"
    _function_flow_function: "KernelFunction"

    def __init__(self, kernel: Kernel, config: SequentialPlannerConfig = None, prompt: str = None):
        assert isinstance(kernel, Kernel)
        self.config = config or SequentialPlannerConfig()

        self.config.excluded_plugins.append(self.RESTRICTED_PLUGIN_NAME)

        self._function_flow_function = self._init_flow_function(prompt, kernel)

        self._context = kernel.create_new_context()

    def _init_flow_function(self, prompt: str, kernel: Kernel):
        prompt_config = PromptTemplateConfig.from_json(read_file(PROMPT_CONFIG_FILE_PATH))
        prompt_template = prompt or read_file(PROMPT_TEMPLATE_FILE_PATH)
        prompt_config.execution_settings.extension_data["max_tokens"] = self.config.max_tokens

        prompt_template = PromptTemplate(
            template=prompt_template,
            template_engine=kernel.prompt_template_engine,
            prompt_config=prompt_config,
        )
        function_config = SemanticFunctionConfig(prompt_config, prompt_template)

        return kernel.register_semantic_function(
            plugin_name=self.RESTRICTED_PLUGIN_NAME,
            function_name=self.RESTRICTED_PLUGIN_NAME,
            function_config=function_config,
        )

    async def create_plan(self, goal: str) -> Plan:
        if len(goal) == 0:
            raise PlanningException(PlanningException.ErrorCodes.InvalidGoal, "The goal specified is empty")

        relevant_function_manual = await KernelContextExtension.get_functions_manual(self._context, goal, self.config)
        self._context.variables.set("available_functions", relevant_function_manual)

        self._context.variables.update(goal)

        plan_result = await self._function_flow_function.invoke(context=self._context)

        if plan_result.error_occurred:
            raise PlanningException(
                PlanningException.ErrorCodes.CreatePlanError,
                f"Error creating plan for goal: {plan_result.last_error_description}",
                plan_result.last_exception,
            )

        plan_result_string = plan_result.result.strip()

        try:
            get_plugin_function = self.config.get_plugin_function or SequentialPlanParser.get_plugin_function(
                self._context
            )
            plan = SequentialPlanParser.to_plan_from_xml(
                plan_result_string,
                goal,
                get_plugin_function,
                self.config.allow_missing_functions,
            )

            if len(plan._steps) == 0:
                raise PlanningException(
                    PlanningException.ErrorCodes.CreatePlanError,
                    (
                        "Not possible to create plan for goal with available functions.\n",
                        f"Goal:{goal}\nFunctions:\n{relevant_function_manual}",
                    ),
                )

            return plan

        except PlanningException as e:
            if e.error_code == PlanningException.ErrorCodes.CreatePlanError:
                raise e
            elif e.error_code in [
                PlanningException.ErrorCodes.InvalidPlan,
                PlanningException.ErrorCodes.InvalidGoal,
            ]:
                raise PlanningException(
                    PlanningException.ErrorCodes.CreatePlanError,
                    "Unable to create plan",
                    e,
                )
            else:
                raise PlanningException(
                    PlanningException.ErrorCodes.CreatePlanError,
                    "Unable to create plan",
                    e,
                )

        except Exception as e:
            raise PlanningException(
                PlanningException.ErrorCodes.UnknownError,
                "Unknown error creating plan",
                e,
            )
