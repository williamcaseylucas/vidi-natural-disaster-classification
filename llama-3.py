# MAIN ENTRYPOINT

# https://huggingface.co/docs/transformers/en/tasks/question_answering

import warnings

warnings.filterwarnings("ignore")


from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM  # text

# from langchain_ollama import ChatOllama  # for chat
# from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)  # may want to swap this out with ConversationChain but yuck
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
                # max frame window is 6
                video_caption_generator = GitCaptioner(
                    with_classification=with_classification
                )
            case "timesformer":
                # max frame window is 8
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
                "Your job is to answer questions related to a central natural disaster that happened. Whatever natural disaster occurs the most often, assume that is the central theme. Answer questions related to the context you are provided.",
                "If you think things are random and not relevant, do not mention them.",
                "I am going to provide a collection of starting times, ending times, summaries of that interval of time, and a classificaition prediction of what happened in that interval of time. The 'Timestamp' represents what part of the video the description is related to. The 'Description' is a summary of what happened in that interval of time. Keep in mind that this 'Description' may not always be accurate so try to do your best to make sense of it. The 'Classification' is a predicted category of what type of disaster may have occurred. Keep in mind sometimes the 'Classification' will not always be accurate.",
                "Be sure to note if you see information related to debris or damage to infrastructure such as buildings, highways, etc.",
                "All of the information you will be provided is going to be related to a video where a natural disaster has occurred so it is your job to try and be as helpful as possible and to make connections between the information you recieve and provide to the user. Some of the captions may be incorrect so if some of them don't seem to follow a consistent flow, ignore them. Also use your reasoning to determine when classifications are incorrect. For example, if a volanic eruption is described as a nuclear explosion, assume the classification is incorrect.",
                "Do not be overly wordy unless you are asked to. Be conversational but answer the question directly.",
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
                self.template(
                    interval["start"],
                    interval["end"],
                    interval["caption"],
                    interval["classification"],
                )
                for interval in captions
            ],
        )
        follow_up = "\nBased on this information, introduce yourself briefly and give a brief summary of the video but do not include information that seems like it could be incorrect. Keep in mind these caption and classifications are predictions and not ground truth values. Then show the user a couple of sample questions the user can ask you."
        res = self.conversation.predict(
            input=user_input + follow_up
        )  # can see res but don't need to
        print("-------------[AI]-------------")
        print(res)
        self.establish_context = True

    def get_video_caption_generator(self):
        return self.video_caption_generator

    def get_captions_from_video(self, video_path, interval_of_window):
        captions = self.video_caption_generator.get_captions_and_intervals_in_seconds(
            video_path=video_path,
            interval_of_window=interval_of_window,
        )
        return captions

    def ask_question(self, question, verbose=True):
        assert (
            self.establish_context
        ), "Context has not been established. Pass captions with start and end timestamps into establish_context() first."

        res = self.conversation.predict(input=question)
        if verbose:
            print(res)
        return res


def get_paths_to_videos(file_path):
    assert file_path, "file_path is required"

    paths = []
    for root, dirs, files in os.walk(file_path):
        if not files:
            continue
        for file in files:
            paths.append(os.path.join(root, file))
    return paths


# load video, don't include .DS_Store
video_paths = list(
    filter(
        lambda path: path.split(".")[-1] != "DS_Store",
        get_paths_to_videos("./dataset/full_video_examples"),
    )
)
video_path = video_paths[2]
# if video_path.split(".")[-1] == "mp4":
#     demo = Video.from_file(video_path)  # to view video
#     display(demo)
natural_disaster = video_path.split("/")[-2]

chat = Llama3Chat(
    video_caption_generator=VideoCaptionType.GIT,
    verbose=False,
    with_classification=True,
)
print("\n\n")
print("Natural disaster label from video: ", natural_disaster)
print("Video path: ", video_path)
captions = chat.get_captions_from_video(video_path=video_path, interval_of_window=10)
chat.establish_context(captions=captions)

while True:
    print("-------------[User]-------------")
    print("[type q, quit, or exit to exit terminal]")
    user_input = input("> ")
    if user_input in ["exit", "q", "quit"]:
        break
    print("")
    print("-------------[AI]-------------")
    chat.ask_question(question=user_input)
