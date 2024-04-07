import re
from enum import Enum
from typing import Annotated
import dspy
from dspy.predict import Retry
from dspy.functional import TypedChainOfThought
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
import typer

app = typer.Typer()


class LanguageModel(str, Enum):
    M_0_5 = "qwen:0.5b"
    M_1_8 = "qwen:latest"
    M_7_A = "meditron:7b"
    M_7_B = "mistral:v0.2"
    M_13 = "mixtral:latest"
    M_35 = "command-r:latest"
    M_70 = "meditron:70b"


class StepType(Enum):
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
    step_annotation: str = dspy.OutputField(
        desc=f"""Must be one of the following values: {
            [item.value for item in StepType]}"""
    )


class MessageWithUnderstanding(dspy.BaseModel):
    clear_rephrasing_of_message: str = dspy.Field(
        description="Rephrase the user's question in clearer form. Leave it empty unless a rephrasing is useful."
    )
    why_is_user_asking_this: str = dspy.Field(
        description="Why is the user asking this question at this point in the ongoing chat?"
    )
    what_is_user_objective: str = dspy.Field(
        description="What are user's overall objectives implicit or explicit within this chat?"
    )
    question_decomposition: list[str] = dspy.Field(
        description="Break down the question into simpler sub-questions."
    )


class UnderstandMessage(dspy.Signature):
    chat: list[str] = dspy.InputField(desc="The conversational history till now")
    new_message: str = dspy.InputField(desc="A new message by the user")
    structured_message: MessageWithUnderstanding = dspy.OutputField(
        desc="Message understood in terms of the underlying intent and objective"
    )


class ConversationalResponse(dspy.Signature):
    raw_message_from_user: str = dspy.InputField()
    structured_message: str = dspy.InputField()
    rationale: str = dspy.InputField(
        desc="Rationale behind the conversational response"
    )
    response_to_user: str = dspy.OutputField(
        desc="Response to the user in a conversational style"
    )


class Task(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, structured_message: MessageWithUnderstanding) -> dspy.Prediction:
        answer = self.generate(question=str(structured_message))
        return answer


def rationale_to_steps(rationale: str) -> list[str]:
    pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    sentences = re.split(pattern, rationale)
    # print(sentences)
    return sentences


# class VerificationStrategy(Enum, str):
#     LLM_AS_A_JUDGE = "llm_as_a_judge"
#     CAPPY = "cappy"
#     RM_MODEL = "rm_model"
#     ROSCOE = "roscoe"
#
#
# class StepVerifier:
#     def __init__(self, strategy: VerificationStrategy):
#         match strategy:
#             case strategy.LLM_AS_A_JUDGE:
#                 pass
#
#     def verify_steps(self, steps: list[str]):
#         pass
#


class VerifiedQA(dspy.Module):
    def __init__(self, verifier_model):
        super().__init__()
        assert isinstance(
            verifier_model, dspy.OllamaLocal
        ), "Currently, a verifier model must be an ollama local model."
        self.verifier_model = verifier_model
        self.question_understanding = TypedChainOfThought(UnderstandMessage)
        self.task = Task()
        self.step_verifier = TypedChainOfThought(StepVerfication)
        self.converational = TypedChainOfThought(ConversationalResponse)

    def forward(self, question: str) -> dspy.Prediction:
        structured_message: MessageWithUnderstanding = self.question_understanding(
            chat=[], new_message=question
        ).structured_message
        # print(structured_message)

        answer = self.task(structured_message)
        steps = rationale_to_steps(answer.rationale)

        with dspy.context(lm=self.verifier_model):
            previous_step = ""
            for step in steps:
                dspy.Suggest(
                    result=self.step_verifier(
                        objective=structured_message.what_is_user_objective,
                        question=question,
                        previous_step=previous_step,
                        current_step=step,
                    ).step_annotation
                    != StepType.NECESSARY_ESSENTIAL_AND_VALID.value,
                    msg="Each step in the thought process must be necessary for reaching an answer, and be logically and factually valid.",
                )
                print("Suggest Failed!")
                previous_step = step

        return self.converational(
            raw_message_from_user=question,
            structured_message=str(structured_message),
            rationale=answer.answer,
        )


@app.command()
def chat(
    question: str,
    debug: Annotated[
        bool, typer.Option(help="If debug, values should be printed to stdout.")
    ] = False,
    model: Annotated[
        LanguageModel, typer.Option(help="Name of one of the local models.")
    ] = LanguageModel.M_7_B,
):
    lm = dspy.OllamaLocal(model=model.value)

    with dspy.context(lm=lm, trace=[]):
        agent = assert_transform_module(
            VerifiedQA(
                verifier_model=dspy.OllamaLocal(model="command-r:latest")
            ).map_named_predictors(Retry),
            backtrack_handler,
        )
        response = agent(question)

        print(response.response_to_user)

    if debug:
        print(lm.inspect_history(n=5))


if __name__ == "__main__":
    app()
