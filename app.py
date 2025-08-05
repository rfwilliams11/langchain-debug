import sqlite3
import requests
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

load_dotenv(dotenv_path="./.env", override=True)


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
        """
        SELECT Album.Title, Artist.Name 
        FROM Album 
        JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        WHERE Artist.Name LIKE :artist;
        """,
        parameters={"artist": f"%{artist}%"},
        include_columns=True,
    )


@tool
def get_tracks_by_artist(artist: str):
    """Get songs by an artist (or similar artists)."""
    return db.run(
        """
        SELECT Track.Name as SongName, Artist.Name as ArtistName 
        FROM Album 
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
        WHERE Artist.Name LIKE :artist;
        """,
        parameters={"artist": f"%{artist}%"},
        include_columns=True,
    )


@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    return db.run(
        """
        SELECT * FROM Track WHERE Name LIKE :song_title;
        """,
        parameters={"song_title": f"%{song_title}%"},
        include_columns=True,
    )


@tool
def get_customer_info(first_name: str, last_name: str):
    """Return basic customer information and song list purchase history"""
    return db.run(
        """
        SELECT Customer.FirstName as first_name, Customer.LastName as last_name, Track.Name as song_title, Artist.Name as artist_name, Album.Title as album_title
        FROM Customer
        LEFT JOIN Invoice ON Invoice.CustomerId = Customer.CustomerId
        LEFT JOIN InvoiceLine ON InvoiceLine.InvoiceId = Invoice.InvoiceId
        LEFT JOIN Track ON Track.TrackId = InvoiceLine.TrackId
        LEFT JOIN Album ON Album.AlbumId = Track.AlbumId
        LEFT JOIN Artist ON Artist.ArtistId = Album.ArtistId
        WHERE Customer.FirstName = :first_name AND Customer.LastName = :last_name;
        """,
        parameters={"first_name": first_name, "last_name": last_name},
        include_columns=True,
    )


system_message = """Your job is to help as a customer service representative for a music store. 

You are helping customers find songs they are looking for or provide recommendations for music you think they would like based on information provided.

You should interact politely with customers to try to figure out how you can help. Ask questions if you need more information.

You can help in a few ways:

- Recommending music: if a customer wants to find some music or information about music
- Purchase history: provide a customer with their purchase history
- Inventory check: letting the customer know what music is in the music store's inventory

You have access to tools to check in the music stores inventory database.

IMPORTANT: Only recommend music that the music store has in inventory. Use the tools to check this.

Use the tools available to you to help answer the customer's questions.
Otherwise, respond."""


# create state
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


tools = [get_albums_by_artist, get_tracks_by_artist, check_for_songs, get_customer_info]
model_with_tools = model.bind_tools(tools)


# create nodes
def model_node(state: AgentState):
    messages_with_system = [SystemMessage(content=system_message)] + state["messages"]

    # Return only the new message - LangGraph will append it to state
    return {"messages": [model_with_tools.invoke(messages_with_system)]}


tools_node = ToolNode(tools)

# create graph
builder = StateGraph(AgentState)
builder.add_node("model", model_node)
builder.add_node("tools", tools_node)

builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()
