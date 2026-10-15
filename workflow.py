from langgraph.graph import END, START, StateGraph

from agents import response_agent, retriever_agent, rewritten_query_agent
from models import State


class Workflow:
    def __init__(self):
        self.rewritten_query = rewritten_query_agent
        self.retriever_agent = retriever_agent
        self.response_agent = response_agent
        self.graph = self.build()

    def build(self):
        graph = StateGraph(State)

        graph.add_node("rewritten_query_agent", self.rewritten_query)
        graph.add_node("retriever_agent", self.retriever_agent)
        graph.add_node("response_agent", self.response_agent)

        graph.add_edge(START, "rewritten_query_agent")
        graph.add_edge("rewritten_query_agent", "retriever_agent")
        graph.add_edge("retriever_agent", "response_agent")
        graph.add_edge("response_agent", END)

        return graph.compile()

    def run(self, initial_state: State):
        return self.graph.invoke(initial_state)
