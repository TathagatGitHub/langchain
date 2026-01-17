from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings   

#1. Set up the API Key 
google_api_key = "AIzaSyBSkcnZpIuNNkHh_y5KfDSN7qBlHKpaeM8"

#2. Initialize the model
model = GoogleGenerativeAI(api_key=google_api_key, model="models/text-embedding-004")


#7. Initialize the embeddings
embeddings = GoogleGenerativeAIEmbeddings(api_key=google_api_key, model="models/text-embedding-004")

#8. Embed a query (Turn text into a Vector)
text = "What is a LangChain Agent?"
vector = embeddings.embed_query(text)

# 9. Look at the result
print(f"Original Text: '{text}'")
print(f"Vector Length: {len(vector)}")  # Likely 768 dimensions
print(f"First 5 numbers: {vector[:5]}") # The first 5 coordinates
