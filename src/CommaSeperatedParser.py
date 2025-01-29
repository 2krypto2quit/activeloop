from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
import os

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI object
model = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the CommaSeparatedListOutputParser
parser = CommaSeparatedListOutputParser()

# Prepare the Prompt
template = """
Offer substitute words for '{target_word}' based on this context: {context}.
Give me a comma-separated list of words only, no explanations.
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# Call the model with the formatted input
output = model.invoke(model_input)

# Extract the content from the AIMessage object and parse it
try:
    parsed_output = parser.parse(output.content)
    print("Parsed Output:", parsed_output)
except Exception as e:
    print("Error parsing output:", e)