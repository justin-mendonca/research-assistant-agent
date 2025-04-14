from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import PyPDFLoader
import os
from datetime import datetime

load_dotenv(override=True)

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# ResearchResponse class for processing chunks
class ChunkResponse(BaseModel):
    summary: str

# Response class for final summary
class ResearchResponse(BaseModel):
    title: str
    summary: str
    references: list[str]

research_response_parser = PydanticOutputParser(pydantic_object=ResearchResponse)
chunk_parser = PydanticOutputParser(pydantic_object=ChunkResponse)

# Update prompt template for processing PDF chunks
chunk_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a research assistant analyzing sections of an academic paper.
        Extract and summarize the key information from the provided text section.
        Focus on findings, methodologies, and implications if present in this section.
        \n{format_instructions}
        """
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{research_paper}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=chunk_parser.get_format_instructions())

# Prompt for final summary
final_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a research assistant tasked with creating a coherent summary of an academic paper.
        Based on the provided section summaries, create a single, flowing summary that captures
        the key points of the entire paper. Do not simply concatenate the summaries.
        
        Your summary should:
        1. Identify and include the paper's title
        2. Create a coherent narrative that flows naturally
        3. Highlight the main contributions and findings
        4. Include any important references mentioned
        
        \n{format_instructions}
        """
    ),
    ("human", "{summaries}")
]).partial(format_instructions=research_response_parser.get_format_instructions())

research_assistant_agent = create_tool_calling_agent(
    llm=llm,
    prompt=chunk_prompt,
    tools=[]
)

agent_executor = AgentExecutor(
    agent=research_assistant_agent,
    verbose=True,
    tools=[]
)

def validate_pdf_file(file_path):
    """Validate that the PDF file exists and is accessible"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if not file_path.lower().endswith('.pdf'):
        raise ValueError("File must be a PDF")
    return True

def save_summary_to_file(summary, input_pdf_path):
    """Save the summary to a text file with timestamp"""
    # Create output directory if it doesn't exist
    output_dir = "summaries"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename based on input PDF name and timestamp
    pdf_name = os.path.basename(input_pdf_path).replace('.pdf', '')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/{pdf_name}_summary_{timestamp}.txt"
    
    # Write summary to file
    with open(output_file, 'w') as f:
        f.write(f"Summary of: {pdf_name}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 80 + "\n\n")
        f.write(summary)
    
    return output_file

def generate_final_summary(summaries):
    """Generate a final coherent summary from all chunks"""
    try:
        print("Generating final summary...")
        # Create a single input from all summaries
        combined_input = "Here are the section summaries of the paper:\n\n" + "\n\n".join(summaries)
        
        # Use the final prompt to generate coherent summary
        final_response = llm.invoke(
            final_prompt.format(summaries=combined_input)
        )
        
        # The response will be in the format specified by ResearchResponse
        final_parsed = research_response_parser.parse(final_response.content)
        
        return final_parsed
    except Exception as e:
        print(f"Error in final summary generation: {e}")
        return None

def main(pdf_path):
    try:
        validate_pdf_file(pdf_path)
        
        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split()
        
        if not docs:
            raise ValueError("No content found in PDF")
            
        print(f"Processing PDF: {pdf_path}")
        print(f"Number of chunks: {len(docs)}")
        
        # Process each chunk and summarize
        chunk_summaries = []
        for i, chunk in enumerate(docs):
            try:
                print(f"Processing chunk {i + 1}/{len(docs)}...")
                raw_response = agent_executor.invoke({"research_paper": chunk.page_content})
                parsed_chunk = chunk_parser.parse(raw_response.get("output")[0]["text"])
                chunk_summaries.append(parsed_chunk.summary)
            except Exception as e:
                print(f"Error processing chunk {i + 1}: {e}")
                continue

        # Generate final summary
        if chunk_summaries:
            final_summary = generate_final_summary(chunk_summaries)
            
            if final_summary:
                # Format the final output
                output_text = f"""Title: {final_summary.title}

Summary:
{final_summary.summary}

References:
{chr(10).join('- ' + ref for ref in final_summary.references)}
"""
                # Save to file
                output_file = save_summary_to_file(output_text, pdf_path)
                print(f"\nFinal summary saved to: {output_file}")
            else:
                # If final summary generation failed, save the intermediate summaries
                output_file = save_summary_to_file("\n\n".join(chunk_summaries), pdf_path)
                print(f"\nIntermediate summaries saved to: {output_file}")
        else:
            print("No summaries were generated. Please check the input PDF and try again.")
            
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return False
    
    return True

if __name__ == "__main__":
    pdf_path = "./pdfs/test.pdf"  # You can modify this or accept as command line argument
    main(pdf_path)