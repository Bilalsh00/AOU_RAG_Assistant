from typing import Any, Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    user_input: str
    chat_history: Annotated[list, add_messages]
    context: Optional[list[Any]]
    sources: Optional[list[str]]
    rewritten_query: Optional[str]
    response: Optional[str]
