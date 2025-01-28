from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided, answer
with "I don't know".
Context: Quantum computing is an emerging field that leverages quantum mechanics to solve complex problems faster than classical computers.
...
Question: {query}
Answer: """

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)

# Create the chain for the prompt
chain = prompt_template | llm 

# Set the query you want to ask
input_data = {"query": "What is the main advantage of quantum computing over classical computing?"}

# Run the chain to get the AI-generated answer
response = chain.invoke(input_data)

print("Question:", input_data["query"])
print("Answer:", response.content)