import sqlite3
import requests
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# Load in memory db
def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)


# Load model
model = ChatOpenAI(model="gpt-4o")


# Create tools for the agent to use
@tool
def get_albums_by_artist(artist: str):
    """Get albums by an artist."""
    return db.run(
        f"""
        SELECT Album.Title, Artist.Name 
        FROM Album 
        JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True,
    )


@tool
def get_tracks_by_artist(artist: str):
    """Get songs by an artist (or similar artists)."""
    return db.run(
        f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName 
        FROM Album 
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
        WHERE Artist.Name LIKE '%{artist}%';
        """,
        include_columns=True,
    )


@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    return db.run(
        f"""
        SELECT * FROM Track WHERE Name LIKE '%{song_title}%';
        """,
        include_columns=True,
    )


# Create prompts
song_system_message = """Your job is to help a customer find any songs they are looking for. 

You only have certain tools you can use. If a customer asks you to look something up that you don't know how, politely tell them what you can help with.

When looking up artists and songs, sometimes the artist/song will not be found. In that case, the tools will return information \
on simliar songs and artists. This is intentional, it is not the tool messing up."""

system_message = """Your job is to help as a customer service representative for a music store.

You should interact politely with customers to try to figure out how you can help. You can help in a few ways:

- Recomending music: if a customer wants to find some music or information about music. Call the router with `music`

If the user is asking or wants to ask about music, send them to that route.
Otherwise, respond."""


# create state
class AgentState(TypedDict):
    messages: list[AnyMessage]


# create nodes
def model_node(state: AgentState):
    return {"messages": state["messages"] + [model.invoke(state["messages"])]}


def router(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "__end__"
    else:
        return "tools"


tools = [get_albums_by_artist, get_tracks_by_artist, check_for_songs, get_customer_info]
tools_node = ToolNode(tools)

# create graph
builder = StateGraph(AgentState)
builder.add_node("model", model_node)
builder.add_node("tools", tools_node)
builder.add_node("router", router)

builder.add_edge(START, "model")
builder.add_edge("tools", "model")
builder.add_edge("model", "router")
builder.add_edge("router", "tools")
builder.add_edge("model", END)

graph = builder.compile()
