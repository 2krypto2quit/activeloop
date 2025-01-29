from langchain_core.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List
from langchain.output_parsers import OutputFixingParser
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

    # Throw error in case of receiving a numbered-list from API
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

# Test the misformatted output
misformatted_output = '{"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}'

print("Testing misformatted output...")
try:
    parsed_output = parser.parse(misformatted_output)
    print("Parsed Output:", parsed_output)
except Exception as e:
    print("Error parsing output:", e)

# Fix the misformatted output
outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
response = outputfixing_parser.parse(misformatted_output)
print("Fixed Output:", response)