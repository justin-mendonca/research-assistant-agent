from dotenv import load_dotenv
from pydantic_core import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import save_tool

load_dotenv()

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# ResearchResponse class is used to define the sttructure of the output, which is used when instnating the output parser
# The output parser is used to parse the output of the LLM and convert it into a Pydantic model
class ResearchResponse(BaseModel):
    title: str
    summary: str
    references: list[str]
    tools_used: list[str]

# instantiate the output parser that is used to parse the output of the LLM
research_response_parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# instantiate the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that has been tasked with analyzing a research paper that has been published in an academic journal.
            You will be provided a URL to the journal article. Analyze the key findings, methodologies, and implications of the research.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=research_response_parser.get_format_instructions())

# Extend tools with other tools that the agent can use to complete the task!
tools = [save_tool]

research_assistant_agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# instantiate the agent executor, which is used to execute the agent that has been defined
# Verbose is used to see the "thinking" of the agent as it streams
agent_executor = AgentExecutor(
    agent=research_assistant_agent,
    verbose=True,
    tools=tools
)

query = input("Please enter the URL of the research paper you want to analyze: ")

raw_response = agent_executor.invoke({ "query": query })

# The output of the LLM is parsed using the output parser that has been defined
try:
    parsed_response = research_response_parser.parse(raw_response.get("output")[0]["text"])
    print("Parsed response:", parsed_response)
except Exception as e:
    print("Error parsing response:", e)
    print("Raw response:", raw_response)