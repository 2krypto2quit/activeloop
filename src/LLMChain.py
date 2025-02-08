from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain.output_parsers import CommaSeparatedListOutputParser  
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
load_dotenv()

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", 
             api_key=os.getenv("OPENAI_API_KEY"), 
             temperature=0)

#llm chain
prompt_template = "What is a word to replace the following word: {word}?"
chain = PromptTemplate.from_template(prompt_template) | llm
# For single input
response = chain.invoke({"word": "artificial"})
#print(response)

#multiple inputs
input_list = [
    {"word": "artificial"},
    {"word": "intelligence"},
    {"word": "robot"}
]
responses = [] 

for input in input_list:
    response = chain.invoke(input)
    responses.append(response)  

#print(responses)

#template for the prompt - Context-aware chain
prompt_template_2 = "Looking at the context of '{context}'. What is an appropriate word to replace the following: {word}?"

chain2 = PromptTemplate.from_template(prompt_template_2) | llm
response2 = chain2.invoke({"word": "fan", "context": "object"})
#print(response2)

#output parser chain
output_parser = CommaSeparatedListOutputParser()

#formart instructions for the output
format_instructions = "Your response should be a comma-separated list of words only, no explanations."

#template for the prompt
prompt_template_3 = """ List all possible words as substitute for 'artificial' as comma seperated,

format_instructions
"""

prompt = PromptTemplate(
    template=prompt_template_3,
    input_variables=[],
    partial_variables={"format_instructions": format_instructions}
)

chain3 = prompt | llm | output_parser
response3 = chain3.invoke({})
#print(response3)

#Create a conversation chain
history = ChatMessageHistory()

prompt = ChatPromptTemplate.from_messages([
  MessagesPlaceholder(variable_name="chat_history"), (
      "human",
      "{input}")
])

chain = prompt | llm 

response4 = chain.invoke({"chat_history": history.messages, 
                         "input": 
                         "List all possible words as substitute for 'artificial' as comma seperated."
                         })

#add the response to the history
history.add_user_message("List all possible words as substitute for 'artificial' as comma seperated")
history.add_ai_message(response4)

# Print the conversation history
print("\nFinal conversation history:")
for message in history.messages:
    print(f"{message.type}: {message.content}")


