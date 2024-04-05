from datetime import datetime
from zoneinfo import ZoneInfo
from minimal_agent import Tool, Agent, LanguageModel, Task, Crew, ProcessType


class SearchTool(Tool):
    name = "Search"
    description = "Search the internet for information"

    def __call__(self, query, *args, **kwargs):
        # Placeholder for search functionality
        return f"Search results for: {query}"


class CalculatorTool(Tool):
    name = "Calculator"
    description = "Perform mathematical calculations"

    def __call__(self, expression, *args, **kwargs):
        # Placeholder for calculator functionality
        return f"Result of {expression} is: {eval(expression)}"


# Create some agents
agent1 = Agent(
    role="Researcher",
    goal="Find relevant information for the given topic",
    backstory="A diligent researcher with expertise in various fields",
    tools=[SearchTool()],
    llm=LanguageModel.M_7_A,
)

agent2 = Agent(
    role="Analyst",
    goal="Analyze the given data and provide insights",
    backstory="An experienced analyst skilled in data interpretation",
    tools=[CalculatorTool()],
    llm=LanguageModel.M_35,
)

# Create some tasks
task1 = Task(
    description="Research the impact of climate change on biodiversity",
    expected_output="A report summarizing the key findings and implications",
    agent="Researcher",
    created_at=datetime.now(ZoneInfo("UTC")),
)

task2 = Task(
    description="Analyze the sales data for the past quarter",
    expected_output="A summary of the sales trends and insights",
    agent="Analyst",
    created_at=datetime.now(ZoneInfo("UTC")),
)

# Create a crew
crew = Crew(
    tasks=[task1, task2],
    agents=[agent1, agent2],
    process=ProcessType.SEQUENTIAL,
    verbose=True,
)

# Print the crew details
print(crew.json(indent=2))
