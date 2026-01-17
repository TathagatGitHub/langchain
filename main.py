from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI   

#1. Set up the API Key 
google_api_key = "AIzaSyBSkcnZpIuNNkHh_y5KfDSN7qBlHKpaeM8"

#2. Initialize the model
model = GoogleGenerativeAI(api_key=google_api_key, model="gemini-flash-latest")

#3. Define the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

#4. Define the output parser
output_parser = StrOutputParser()

#5. Define the chain
chain = prompt_template | model | output_parser

#6. Run the chain
user_input = input("Enter your question: ")
result = chain.invoke({"input": user_input})
print(result)
