from dotenv import load_dotenv
from pydantic_core import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent

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
