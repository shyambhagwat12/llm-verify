import re
from enum import Enum
import fire
import dspy
from dspy.predict import Retry
from dspy.functional import TypedChainOfThought
from dspy.pydantic import BaseModel, Field

# from dspy.pydantic import BaseModel
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

models = {  # Active param count used as keys
    "0.5": "qwen:0.5b",
    "1.8": "qwen:latest",
    "7:a": "meditron:7b",
    "7:b": "mistral:v0.2",
    "13": "mixtral:latest",
    "35": "command-r:latest",
    "70": "meditron:70b",
}


class StepType(str, Enum):
    NECESSARY_ESSENTIAL_AND_VALID = "necessary_essential_valid"
    UNNECESSARY = "unnecessary"
    LOGICALLY_FALSE = "logically_false"
    NOT_BACKED_BY_PRIOR_FACTS = "not_backed_by_prior_facts"
    BAD_DEDUCTIVE_REASONING = "bad_deductive_reasoning"


class StepVerfication(dspy.Signature):
    objective: str = dspy.InputField()
    question: str = dspy.InputField()
    previous_step: str = dspy.InputField()
    current_step: str = dspy.InputField()
    step_annotation: StepType = dspy.OutputField(
        desc=f"Must be one of the following values: {StepType}"
    )


class Task(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, input: str) -> dspy.Prediction:
        answer = self.generate(question=input)
        return answer


def rationale_to_steps(rationale: str) -> list[str]:
    pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    sentences = re.split(pattern, rationale)
    # print(sentences)
    return sentences


class BaseAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.task = Task()
        self.step_verifier = TypedChainOfThought(StepVerfication)

    def forward(self, question: str) -> dspy.Prediction:
        answer = self.task(question)
        steps = rationale_to_steps(answer.rationale)

        previous_step = ""
        for step in steps:
            dspy.Suggest(
                result=self.step_verifier(
                    objective="Given a message from the user, come up with a reply by thinking step by step",
                    question=question,
                    previous_step=previous_step,
                    current_step=step,
                ).step_annotation
                == StepType.NECESSARY_ESSENTIAL_AND_VALID,
                msg="Each step in the thought process must be correct.",
            )
            previous_step = step

        return answer


class AgentCLI(object):
    """Optimize an agent, evaluate it, chat with it or deploy it."""

    def __init__(self, model: str, debug: bool = False):
        self.debug = bool(debug)
        model = str(model)
        if model not in models.keys():
            raise ValueError(f"Model ({model}) must be one of {models.keys()}")

        self.lm = dspy.OllamaLocal(model=models[model])

    def chat(self, question: str):
        with dspy.settings.context(lm=self.lm, trace=[]):
            agent = assert_transform_module(
                BaseAgent().map_named_predictors(Retry), backtrack_handler
            )
            response = agent(question)

            print(response.answer)

        if self.debug:
            print(self.lm.inspect_history(n=5))


if __name__ == "__main__":
    fire.Fire(AgentCLI)
