import json
import os
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from langchain_core.chat_history import (BaseChatMessageHistory, InMemoryChatMessageHistory)
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from elevenlabs import stream
llm = ChatOllama(model="dolphin-mistral:latest")
load_dotenv()




client = ElevenLabs(
    api_key=os.getenv('ELEVENLABS_API_KEY'),
)

def textToSpeech(input: str):
    audio_stream =client.text_to_speech.convert(
    voice_id="flHkNRp1BlvT73UL6gyz",
        output_format="mp3_22050_32",
        text=input,
        model_id="eleven_flash_v2_5",
        voice_settings=VoiceSettings(
            stability=0.50,
            similarity_boost=0.99,
            style=0.8,
            use_speaker_boost=True,
        )
    )
    return audio_stream

# Function to load configuration from a text file
def load_config(filename: str):
    config = {}
    with open(filename, 'r') as file:
        for line in file:
            key, value = line.strip().split(":", 1)
            config[key.strip()] = value.strip()
    return config

# Load the system prompt and starting message from the text file
config = load_config("travelAgent.txt")
systemPrompt = config.get("system_prompt", "Default system prompt.")
starting_message = config.get("starting_message", "Default starting message.")

humanTemplate = f"{{question}}"
promptTemplate = ChatPromptTemplate.from_messages(
    [
        ('system', systemPrompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", humanTemplate),
    ]
)

chain = promptTemplate | llm

# Persistent storage
STORE_FILE = "chat_memory.json"
store = {}

def save_store_to_file():
    """Save the chat history to a file."""
    serialized_store = {
        session_id: [{"type": type(msg).__name__, "content": msg.content} for msg in history.messages]
        for session_id, history in store.items()
    }
    with open(STORE_FILE, "w") as f:
        json.dump(serialized_store, f)

def load_store_from_file():
    """Load the chat history from a file."""
    try:
        with open(STORE_FILE, "r") as f:
            serialized_store = json.load(f)
        for session_id, messages in serialized_store.items():
            history = InMemoryChatMessageHistory()
            for msg in messages:
                if msg["type"] == "HumanMessage":
                    history.add_message(HumanMessage(content=msg["content"]))
                elif msg["type"] == "AIMessage":
                    history.add_message(AIMessage(content=msg["content"]))
            store[session_id] = history
    except FileNotFoundError:
        pass  # No file exists yet, so start fresh

def getBySessionId(sessionId: str) -> BaseChatMessageHistory:
    if sessionId not in store:
        store[sessionId] = InMemoryChatMessageHistory()
    return store[sessionId]

# Load existing store from file at startup
load_store_from_file()

chainWithHistory = RunnableWithMessageHistory(
    chain,
    getBySessionId,
    input_messages_key="question",
    history_messages_key="history",
)

session_id = "foo"

# Start interaction
initial_result = chainWithHistory.invoke(
    {"question": starting_message},
    config={"configurable": {"session_id": session_id}},
)
print(initial_result.content)
#stream(textToSpeech(initial_result.content))

while True:
    userQuestion = input(">>> ")
    if userQuestion == "quit":
        save_store_to_file()  # Save memory before exiting
        break
    result = chainWithHistory.invoke(
        {"question": userQuestion},
        config={"configurable": {"session_id": session_id}},
    )
    print(result.content)
    #stream(textToSpeech(result.content))


