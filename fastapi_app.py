"""
Step 11 — FastAPI Backend
=========================
Endpoints:
  POST /query      — main query endpoint
  GET  /agents     — list all agents + status
  GET  /confidence — confidence stats from feedback log
  GET  /health     — health check
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json, os

from langgraph_orchestration import run_query

app = FastAPI(title="AI Knowledge Fabric API")


# ============================================================
# Models
# ============================================================
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query:             str
    domain:            str
    domain_confidence: float
    agent_confidence:  float
    low_confidence:    bool
    answer:            str
    retrained:         bool


# ============================================================
# POST /query
# ============================================================
@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):

    result = run_query(request.query)

    return {
        "query":             result["query"],
        "domain":            result["domain"],
        "domain_confidence": result["domain_confidence"],
        "agent_confidence":  result["agent_confidence"],
        "low_confidence":    result["low_confidence"],
        "answer":            result["answer"],
        "retrained":         result["retrained"],
    }


# ============================================================
# GET /agents
# ============================================================
@app.get("/agents")
def get_agents():

    return {
        "agents": [
            {"name": "Legal Agent",   "domain": "legal",   "summarizer": "InfurnusWolf/legal_summarizer",   "reasoner": "microsoft/phi-2"},
            {"name": "Coding Agent",  "domain": "coding",  "summarizer": "InfurnusWolf/coding_summarizer",  "reasoner": "microsoft/phi-2"},
            {"name": "Medical Agent", "domain": "medical", "summarizer": "InfurnusWolf/medical_summarizer", "reasoner": "microsoft/phi-2"},
            {"name": "Finance Agent", "domain": "finance", "summarizer": "InfurnusWolf/finance_summarizer", "reasoner": "microsoft/phi-2"},
        ]
    }


# ============================================================
# GET /confidence
# Returns confidence stats per domain
# ============================================================
@app.get("/confidence")
def get_confidence():

    if not os.path.exists("feedback_dataset.json"):
        return {"message": "No queries yet", "stats": {}}

    with open("feedback_dataset.json", "r") as f:
        data = json.load(f)

    stats = {}

    for domain in ["legal", "coding", "medical", "finance"]:

        domain_entries = [e for e in data if e["domain"] == domain]

        if not domain_entries:
            continue

        confidences  = [e["confidence"] for e in domain_entries]
        low_conf     = [e for e in domain_entries if e["needs_review"]]
        retrained    = [e for e in domain_entries if e.get("was_retrained")]

        stats[domain] = {
            "total_queries":        len(domain_entries),
            "avg_confidence":       round(sum(confidences) / len(confidences), 4),
            "min_confidence":       round(min(confidences), 4),
            "max_confidence":       round(max(confidences), 4),
            "low_confidence_count": len(low_conf),
            "retrained_count":      len(retrained),
        }

    return {"stats": stats}


# ============================================================
# GET /health
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# Run with:
#   uvicorn fastapi_app:app --reload
# ============================================================