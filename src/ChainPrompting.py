from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Prompt 1
template_question = """What is the name of the famous scientist who developed the theory of general relativity?
Answer: """
prompt_question = PromptTemplate(template=template_question, input_variables=[])

# Prompt 2
template_fact = """Provide a brief description of {scientist}'s theory of general relativity.
Answer: """
prompt_fact = PromptTemplate(input_variables=["scientist"], template=template_fact)

# Create the Runnable for the first prompt
chain_question = prompt_question | llm

response_question = chain_question.invoke({})

# Extract the scientist's name from the response
scientist = response_question.content.strip()

# Create the Runnable for the second prompt
chain_fact = prompt_fact | llm

# Input data for the second prompt
input_data = {"scientist": scientist}

# Run the Runnable for the second prompt
response_fact = chain_fact.invoke(input_data)

print("Scientist:", scientist)
print("Fact:", response_fact.content)