from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # reads your API key from the .env file

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

reply = llm.invoke(
    "In one simple sentence, what is an auto-renewal clause in a contract?"
)
print(reply.text)