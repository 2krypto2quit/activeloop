from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List
from langchain.output_parsers import RetryWithErrorOutputParser
import os

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI object
model = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitute words based on context")
    reason: List[str] = Field(description="the reasoning of why this word fits this context")

    @field_validator('words')
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return field
    
    @field_validator('reason')
    def end_with_dot(cls, field):
        for idx, item in enumerate(field):
            if item[-1] != ".":
                field[idx] += "."
        return field

parser = PydanticOutputParser(pydantic_object=Suggestions)

# Prepare the Prompt
template = """
Please provide substitute words for '{target_word}' based on this context: {context}

For each substitute word, provide a reason why it fits the context.
{format_instructions}

Make sure each reason ends with a period and words don't start with numbers.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# First try with misformatted output
misformatted_output = '{"words": ["conduct", "manner"]}'

print("Testing misformatted output...")
try:
    parsed_output = parser.parse(misformatted_output)
    print("Parsed Output:", parsed_output)
except Exception as e:
    print("Error parsing output:", e)

# Create retry parser
retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)

# Get response from model and try to parse it
output = model.invoke(model_input.to_string())
try:
    # Try to parse the original output
    parsed_output = parser.parse(output.content)
    print("\nOriginal Parsed Output:", parsed_output)
except Exception as e:
    print("\nError with original parse:", e)
    # If original parse fails, try with retry parser
    fixed_output = retry_parser.parse_with_prompt(output.content, prompt)
    print("\nFixed Output:", fixed_output)