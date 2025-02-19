from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequenc of input messages.
# HumanMessagse:
#   Message from a human to the AI model.
messages = [
    SystemMessage(content="按我说的做"),
    HumanMessage(content="说你爱我"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")


# AIMessage:
#   Message from an AI.
messages = [
    SystemMessage(content="按我说的做"),
    HumanMessage(content="说你爱我"),
    AIMessage(content=result.content),
    HumanMessage(content="完整的重复上一次你的回答"),
]
print(messages)
# Invoke the model with messages
# result = model.invoke(messages)
# print(f"Answer from AI: {result.content}")
