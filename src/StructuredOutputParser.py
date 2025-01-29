from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
import os

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI object
model = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

response_schemas = [
    ResponseSchema(name="words", description="A substitute word based on context"),
    ResponseSchema(name="reasons", description="the reasoning of why this word fits the context.")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create prompt template
template = """
Suggest a substitute for the word '{target_word}' based on this context: {context}
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Format the prompt with your inputs
model_input = prompt.format(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# Get the output from the model
output = model.invoke(model_input)

# Parse the structured output
try:
    parsed_output = parser.parse(output.content)
    print("Parsed Output:", parsed_output)
except Exception as e:
    print("Error parsing output:", e)