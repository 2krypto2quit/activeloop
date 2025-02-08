from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", 
             api_key=os.getenv("OPENAI_API_KEY"), 
             temperature=0)

# poet
poet_template: str = """You are an American poet, your job is to come up with\
poems based on a given theme.

Here is the theme you have been asked to generate a poem on:
{input}\
"""

poet_prompt_template = PromptTemplate(
    input_variables=["input"], template=poet_template)

# Create poet chain using pipe operator
poet_chain = poet_prompt_template | llm

# critic
critic_template: str = """You are a critic of poems, you are tasked\
to inspect the themes of poems. Identify whether the poem includes romantic expressions or descriptions of nature.

Your response should be in the following format, as a Python Dictionary.
poem: this should be the poem you received 
Romantic_expressions: True or False
Nature_descriptions: True or False

Here is the poem submitted to you:
{poem}\
"""

critic_prompt_template = PromptTemplate(
    input_variables=["poem"], template=critic_template)

# Create critic chain using pipe operator
critic_chain = critic_prompt_template | llm

# Create sequential chain
chain = RunnableSequence(
    first=poet_chain,
    last=critic_chain,
)

# Run the chain
result = chain.invoke({"input": "Write a poem about a sunset"})
print("\nPoem:")
print(result)