"""
Step 10 — LangGraph Multi-Agent Orchestration
=============================================
Graph flow:
  User → Manager → Domain Agent → Confidence Evaluator → Self Learning Loop

Install:
  pip install langgraph langchain
"""

import json
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from legal_agent   import LegalAgent
from coding_agent  import CodingAgent
from medical_agent import MedicalAgent
from finance_agent import FinanceAgent
from self_learning import FeedbackLoop
from transformers  import pipeline


# ============================================================
# State — passed between all nodes in the graph
# ============================================================
class AgentState(TypedDict):
    query:             str
    domain:            str
    domain_confidence: float
    agent_confidence:  float
    answer:            str
    low_confidence:    bool
    retrained:         bool


# ============================================================
# Initialize all agents once
# ============================================================
print("Loading agents for LangGraph...")

legal_agent   = LegalAgent()
coding_agent  = CodingAgent()
medical_agent = MedicalAgent()
finance_agent = FinanceAgent()
feedback      = FeedbackLoop()

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

CANDIDATE_LABELS   = ["coding", "legal", "medical", "finance"]
CONFIDENCE_CUTOFF  = 0.60

print("All agents loaded\n")


# ============================================================
# NODE 1 — Manager Agent
# Classifies the query domain
# ============================================================
def manager_node(state: AgentState) -> AgentState:

    print(f"\n[Manager] Classifying: {state['query']}")

    result = classifier(
        state["query"],
        candidate_labels=CANDIDATE_LABELS
    )

    state["domain"]            = result["labels"][0]
    state["domain_confidence"] = round(result["scores"][0], 4)

    print(f"[Manager] Domain: {state['domain']} ({state['domain_confidence']})")

    return state


# ============================================================
# NODE 2 — Domain Agent Router
# Routes to the correct expert agent
# ============================================================
def domain_agent_node(state: AgentState) -> AgentState:

    domain = state["domain"]

    print(f"\n[DomainAgent] Running {domain} agent...")

    agent_map = {
        "coding":  coding_agent,
        "legal":   legal_agent,
        "medical": medical_agent,
        "finance": finance_agent,
    }

    result = agent_map[domain].run(state["query"])

    if isinstance(result, dict):
        state["answer"]           = result["answer"]
        state["agent_confidence"] = result["confidence"]
    else:
        state["answer"]           = result
        state["agent_confidence"] = state["domain_confidence"]

    print(f"[DomainAgent] Confidence: {state['agent_confidence']}")

    return state


# ============================================================
# NODE 3 — Confidence Evaluator
# Decides if answer is reliable
# ============================================================
def confidence_evaluator_node(state: AgentState) -> AgentState:

    confidence = state["agent_confidence"]

    state["low_confidence"] = confidence < CONFIDENCE_CUTOFF

    if state["low_confidence"]:
        print(f"\n[ConfidenceEvaluator] LOW confidence ({confidence}) — flagging")
    else:
        print(f"\n[ConfidenceEvaluator] Confidence OK ({confidence})")

    return state


# ============================================================
# NODE 4 — Self Learning Loop
# Stores query, retrains if threshold reached
# ============================================================
def self_learning_node(state: AgentState) -> AgentState:

    print(f"\n[SelfLearning] Storing query...")

    feedback.store(
        domain=state["domain"],
        query=state["query"],
        context="",
        answer=state["answer"],
        confidence=state["agent_confidence"]
    )

    retrained = feedback.retrain_if_ready(state["domain"])

    state["retrained"] = retrained

    return state


# ============================================================
# CONDITIONAL EDGE
# After confidence eval — always goes to self learning
# (both high and low confidence queries are stored)
# ============================================================
def route_after_confidence(state: AgentState) -> Literal["self_learning", END]:
    return "self_learning"


# ============================================================
# Build the LangGraph
# ============================================================
def build_graph():

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("manager",              manager_node)
    graph.add_node("domain_agent",         domain_agent_node)
    graph.add_node("confidence_evaluator", confidence_evaluator_node)
    graph.add_node("self_learning",        self_learning_node)

    # Add edges
    graph.set_entry_point("manager")
    graph.add_edge("manager",              "domain_agent")
    graph.add_edge("domain_agent",         "confidence_evaluator")
    graph.add_conditional_edges(
        "confidence_evaluator",
        route_after_confidence,
        {"self_learning": "self_learning"}
    )
    graph.add_edge("self_learning", END)

    return graph.compile()


# ============================================================
# Main app — this replaces manager_agent.py
# ============================================================
app = build_graph()


def run_query(query: str) -> dict:

    initial_state: AgentState = {
        "query":             query,
        "domain":            "",
        "domain_confidence": 0.0,
        "agent_confidence":  0.0,
        "answer":            "",
        "low_confidence":    False,
        "retrained":         False,
    }

    final_state = app.invoke(initial_state)

    return final_state


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":

    result = run_query("What is insider trading?")

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Domain     : {result['domain']}")
    print(f"Confidence : {result['agent_confidence']}")
    print(f"Answer     : {result['answer'][:300]}...")