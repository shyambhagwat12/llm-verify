import json
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
import typer
import dspy
from typing_extensions import Annotated
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional, Callable, Union

# import re
# from dspy.predict import Retry
# from dspy.functional import TypedChainOfThought
# from dspy.primitives.assertions import assert_transform_module, backtrack_handler

app = typer.Typer()


class LanguageModel(str, Enum):
    M_0_5 = "qwen:0.5b"
    M_1_8 = "qwen:latest"
    M_7_A = "meditron:7b"
    M_7_B = "mistral:v0.2"
    M_13 = "mixtral:latest"
    M_35 = "command-r:latest"
    M_70 = "meditron:70b"


class Tool(BaseModel):
    name: str
    description: Optional[str] = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Tool must implement the __call__ method")


class Agent(BaseModel):
    role: str
    goal: str
    backstory: str
    tools: List[Tool] = Field(default_factory=list)
    llm: Optional[LanguageModel] = None
    function_calling_llm: Optional[LanguageModel] = None
    max_iter: int = 15
    max_rpm: Optional[int] = None
    verbose: bool = True
    allow_delegation: bool = True
    step_callback: Optional[Callable] = None
    memory: bool = True

    class Config:
        arbitrary_types_allowed = True


class Task(BaseModel):
    description: str
    expected_output: str
    agent: Optional[str] = None
    tools: Optional[List[str]] = Field(default_factory=list)
    async_execution: Optional[bool] = False
    context: Optional[List["Task"]] = Field(default_factory=list)
    output_json: Optional[Dict[str, Any]] = None
    output_pydantic: Optional[BaseModel] = None
    output_file: Optional[str] = None
    callback: Optional[Callable] = None
    human_input: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(ZoneInfo("UTC")))

    class Config:
        arbitrary_types_allowed = True


class ProcessType(str, Enum):
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    # CONSENSUAL = "consensual"  # Planned for future development


class Process(BaseModel):
    type: ProcessType
    manager_llm: Optional[BaseModel] = None

    @field_validator("manager_llm")
    def manager_llm_required_for_hierarchical(cls, v, values):
        if values["type"] == ProcessType.HIERARCHICAL and v is None:
            raise ValueError("manager_llm is required for hierarchical process")
        return v

    class Config:
        use_enum_values = True


class Crew(BaseModel):
    tasks: List[Task]
    agents: List[Agent]
    process: ProcessType = Field(default=ProcessType.SEQUENTIAL)
    verbose: bool = Field(default=False)
    manager_llm: Optional[LanguageModel] = None
    function_calling_llm: Optional[LanguageModel] = None
    config: Optional[Union[Dict[str, Any], str]] = None
    max_rpm: Optional[int] = None
    language: str = Field(default="English")
    full_output: bool = Field(default=False)
    step_callback: Optional[Callable] = None
    share_crew: bool = Field(default=False)

    @field_validator("config", mode="before")
    def parse_config(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("manager_llm")
    def manager_llm_required_for_hierarchical(cls, v, values):
        if values["process"] == ProcessType.HIERARCHICAL and v is None:
            raise ValueError("manager_llm is required for hierarchical process")
        return v

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


class BaseAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, question: str) -> dspy.Prediction:
        pass


@app.command()
def chat(
    question: str,
    verbose: Annotated[
        bool, typer.Option(help="If debug values should be printed to stdout.")
    ],
    model: Annotated[
        LanguageModel,
        typer.Option(help="Name of one of the local models."),
    ] = LanguageModel.M_0_5,
):
    lm = dspy.OllamaLocal(model=model.value)

    with dspy.settings.context(lm=lm, trace=[]):
        print(question)
        print(lm(question))

    if verbose:
        print(lm.inspect_history(n=5))


if __name__ == "__main__":
    app()
