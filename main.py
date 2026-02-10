from dotenv import load_dotenv
load_dotenv()

from crewai import Crew, Agent, Task, LLM
from crewai_tools import SerperDevTool
import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    domain = input("Enter company domain (e.g., stripe.com): ")

    researcher_cfg = load_yaml("agents/researcher.yaml")
    analyst_cfg = load_yaml("agents/analyst.yaml")
    research_task_cfg = load_yaml("tasks/research_task.yaml")
    analysis_task_cfg = load_yaml("tasks/analysis_task.yaml")

    llm = LLM(
        model="gemini-3-flash-preview",
        provider="gemini"
    )

    search_tool = SerperDevTool()

    researcher = Agent(
        **researcher_cfg,
        tools=[search_tool],
        llm=llm
    )

    analyst = Agent(
        **analyst_cfg,
        llm=llm
    )

    research_task = Task(
        description=research_task_cfg["description"] + f"\nCompany Domain: {domain}",
        expected_output=research_task_cfg["expected_output"],
        agent=researcher
    )

    analysis_task = Task(
        description=analysis_task_cfg["description"],
        expected_output=analysis_task_cfg["expected_output"],
        agent=analyst,
        output_file=analysis_task_cfg["output_file"]
    )

    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        tracing=True,
        verbose=True
    )

    crew.kickoff()

if __name__ == "__main__":
    main()