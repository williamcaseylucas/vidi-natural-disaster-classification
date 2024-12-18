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
from tqdm import tqdm

from caption_models import GitCaptioner, TimesformerCaptioner, VideoCaptionType
import os
from ipywidgets import Video
from IPython.display import display


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


class Llama3Chat:
    def __init__(
        self, video_caption_generator, with_classification=False, verbose=False
    ):
        match video_caption_generator.value:
            case "git":
                video_caption_generator = GitCaptioner(
                    with_classification=with_classification
                )
            case "timesformer":
                video_caption_generator = TimesformerCaptioner(
                    with_classification=with_classification
                )
            case _:
                raise ValueError(
                    "video_caption_generator must be one of ['git', 'timesformer']"
                )
        self.video_caption_generator = video_caption_generator

        model = OllamaLLM(model="llama3.2")

        system_message = " ".join(
            [
                "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.",
                "Ensure the video caption's main theme is about a natural disaster and the damages",
                "Ensure you provide information about damaged infrastructure mainly such as buildings, debris, highways",
            ]
        )
        template_for_llm = """

        Entities:
        {entities}

        Current conversation:
        {history}

        Human: {input}
        AI Assistant:"""
        PROMPT = PromptTemplate(
            input_variables=["entities", "history", "input"],
            template=system_message + template_for_llm,
        )

        self.conversation = ConversationChain(
            llm=model,
            memory=ConversationEntityMemory(llm=model),
            prompt=PROMPT,
            verbose=verbose,
        )

        self.template = (
            lambda start, end, caption, classification: f"Timestamp: {start}-{end} seconds. Description: {caption}{' Classification: ' + classification if classification else ''}"
        )

        self.context_established = False

    def establish_context(self, captions):
        # establish context
        user_input = "\n".join(
            [
                "I am going to provide a summary of a video where a natural disaster occurred. The 'Timestamp' represents what part of the video the description is related to. The 'Description' is a summary of what happened in that interval of time. The 'Classification' is a predicted category of what type of disaster may have occurred. Keep in mind sometimes the 'Classification' will not be accurate. It is provided to help provide additional context.",
                *[
                    self.template(
                        interval["start"],
                        interval["end"],
                        *interval["caption"],
                        interval["classification"],
                    )
                    for interval in captions
                ],
            ]
        )
        follow_up = "\nDon't respond. Just use this information for context when I ask the next question"
        res = self.conversation.predict(
            input=user_input + follow_up
        )  # can see res but don't need to
        self.establish_context = True

    def get_video_caption_generator(self):
        return self.video_caption_generator

    def get_captions_from_video(self, video_path, interval_of_window):
        captions = self.video_caption_generator.get_captions_and_intervals_in_seconds(
            video_path=video_path,
            interval_of_window=interval_of_window,
        )
        return captions

    def ask_question(self, question):
        assert (
            self.establish_context
        ), "Context has not been established. Pass captions with start and end timestamps into establish_context() first."

        res = self.conversation.predict(input=question)
        print(res)


def get_paths_to_videos(file_path):
    assert file_path, "file_path is required"

    paths = []
    for root, dirs, files in os.walk(file_path):
        if not files:
            continue
        for file in files:
            paths.append(os.path.join(root, file))
    return paths


# load video
video_paths = get_paths_to_videos("./dataset/full_video_examples")
video_path = video_paths[1]
if video_path.split(".")[-1] == "mp4":
    demo = Video.from_file(video_path)  # to view video
    display(demo)
natural_disaster = video_path.split("/")[-2]

print("natural_disaster", natural_disaster)
chat = Llama3Chat(
    video_caption_generator=VideoCaptionType.TIMESFORMER,
    verbose=False,
    with_classification=True,
)
captions = chat.get_captions_from_video(video_path=video_path, interval_of_window=5)
chat.establish_context(captions=captions)

chat.ask_question("Introduce yourself.")
while True:
    user_input = input(">")
    if user_input == "exit" or "q" or "quit":
        break
    chat.ask_question(question=user_input)
