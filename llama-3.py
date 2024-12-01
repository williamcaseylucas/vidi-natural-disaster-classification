# https://huggingface.co/docs/transformers/en/tasks/question_answering

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM  # text

# from langchain_ollama import ChatOllama  # for chat
# from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain_core.prompts.prompt import PromptTemplate
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE


def text_generation_example():
    """
    Overall, this is better for automation as it is less chatty
    """
    ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant."), ("human", "{question}")]
    )

    model = OllamaLLM(model="llama3.2")

    user_input = input("> ")

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model

    response = chain.invoke({"question": user_input})

    print(response)


def use_buffer_memory(user_input):
    model = OllamaLLM(model="llama3.2")
    conversation = ConversationChain(
        llm=model, memory=ConversationBufferMemory(), verbose=True
    )
    user_input = "how are you doing?"
    res = conversation.predict(input=user_input)
    return res


def use_entity_memory_with_basic_template(
    user_input: list[str] = [
        "My name is William and I am a software engineer.",
        "What is William?",
    ]
):
    model = OllamaLLM(model="llama3.2")
    conversation = ConversationChain(
        llm=model,
        memory=ConversationEntityMemory(llm=model),
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        verbose=True,
    )

    for input in user_input:
        res = conversation.predict(input=input)
        print(res)


def use_entity_memory_with_custom_system_template(
    user_input: list[str] = [
        "My name is William and I am a software engineer.",
        "What is William?",
    ],
    system_template="",
):
    model = OllamaLLM(model="llama3.2")

    system_message = " ".join(
        [
            "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.",
            system_template,
        ]
    )

    template = """

    Entities:
    {entities}

    Current conversation:
    {history}

    Human: {input}
    AI Assistant:"""
    PROMPT = PromptTemplate(
        input_variables=["entities", "history", "input"],
        template=system_message + template,
    )
    conversation = ConversationChain(
        llm=model,
        memory=ConversationEntityMemory(llm=model),
        prompt=PROMPT,
        verbose=True,
    )

    for input in user_input:
        res = conversation.predict(input=input)
        print(res)


# Basic example
# use_entity_memory_with_custom_system_template(
#     system_template="You are a helpful news anchor who is reporting on natural disaster events",
# )

model = OllamaLLM(model="llama3.2")

system_message = " ".join(
    [
        "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.",
        "Ensure the video caption's main theme is about a natural disaster and the damages",
        "Ensure you provide information about damaged infrastructure mainly such as buildings, debris, highways",
    ]
)

template = """

Entities:
{entities}

Current conversation:
{history}

Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(
    input_variables=["entities", "history", "input"],
    template=system_message + template,
)
conversation = ConversationChain(
    llm=model,
    memory=ConversationEntityMemory(llm=model),
    prompt=PROMPT,
    verbose=True,
)


template = (
    lambda start, end, description: f"Timestamp: {start}-{end} seconds. Description: {description}"
)

template(0, 10, "a tree has fallen down and is obstructed by a car.")
# establish context
user_input = "\n".join(
    [
        "I am going to provide a summary of a video where a natural disaster occurred. The 'Timestamp' represents what part of the video the description is related to. The 'Description' is a summary of what happened in that interval of time.",
        "Timestamp: 0-10 seconds. Description: a tree has fallen down and is obstructed by a car.",
        "Timestamp: 10-20 seconds. Description: a building is collapsing and a fire is spreading.",
        "Timestamp: 20-30 seconds. Description: an emergency vehicle is driving down the road.",
    ]
)
follow_up = "\nDon't respond. Just use this information for context when I ask the next question"
print(user_input)
res = conversation.predict(input=user_input + follow_up)
print(res)

question = "Did a tree fall down at some point in the video?"

res = conversation.predict(input=question)
print(res)

question = "What happened in this video?"
res = conversation.predict(input=question)
print(res)
