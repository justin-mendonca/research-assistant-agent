from langchain_community.tools import PubmedQueryRun
from langchain.tools import Tool
from datetime import datetime

# Create custom tool for writing research agent results to a file
def save_research_results(article_title: str, research_results: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_name = f"{article_title}_{timestamp}.txt"

    # format the research results based on how they should be saved to the .txt file
    formatted_results = f"Title: {article_title}\nResearch Completion Time: {timestamp}\n\n{research_results}"

    with open(file_name, "w") as file:
        file.write(formatted_results)

    print(f"Results written to {article_title}.txt")

save_tool = Tool(
    name="save_research_results",
    func=save_research_results,
    description="Saves the research results to a text file with the article title and timestamp.",
)

