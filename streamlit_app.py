import streamlit as st
import json
import os
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI Knowledge Fabric",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.domain-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-coding  { background: #1e3a5f; color: #60a5fa; }
.badge-legal   { background: #3b1f0a; color: #fb923c; }
.badge-medical { background: #1f1a3b; color: #c084fc; }
.badge-finance { background: #0f3b2b; color: #34d399; }

.answer-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 24px;
    font-size: 0.95rem;
    line-height: 1.7;
    margin-top: 16px;
}

.confidence-bar {
    height: 8px;
    border-radius: 4px;
    background: #1f2937;
    margin-top: 6px;
}

.graph-box {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
}
</style>
""", unsafe_allow_html=True)

DOMAIN_BADGE = {
    "coding":  '<span class="domain-badge badge-coding">⌨ Coding</span>',
    "legal":   '<span class="domain-badge badge-legal">⚖ Legal</span>',
    "medical": '<span class="domain-badge badge-medical">🩺 Medical</span>',
    "finance": '<span class="domain-badge badge-finance">📈 Finance</span>',
}

DOMAIN_COLOR = {
    "coding": "#60a5fa",
    "legal":  "#fb923c",
    "medical":"#c084fc",
    "finance":"#34d399",
}

# ============================================================
# Header
# ============================================================
st.markdown("# 🧠 AI Knowledge Fabric")
st.markdown("Ask anything — routes automatically to the right expert agent.")
st.divider()

# ============================================================
# Layout — query left, graph right
# ============================================================
col_query, col_graph = st.columns([2, 1])

with col_graph:
    st.markdown("#### LangGraph Flow")
    st.markdown("""
<div class="graph-box">
User Query<br>
&nbsp;&nbsp;&nbsp;↓<br>
Manager Agent<br>
(BART classifier)<br>
&nbsp;&nbsp;&nbsp;↓<br>
Domain Agent<br>
(Summarizer + Phi-2)<br>
&nbsp;&nbsp;&nbsp;↓<br>
Confidence Evaluator<br>
&nbsp;&nbsp;&nbsp;↓<br>
Self Learning Loop<br>
(retrain if needed)
</div>
""", unsafe_allow_html=True)

    # Confidence stats from /confidence endpoint
    st.markdown("#### Confidence Stats")
    try:
        stats_resp = requests.get(f"{API_URL}/confidence", timeout=5)
        stats      = stats_resp.json().get("stats", {})

        if stats:
            for domain, s in stats.items():
                color = DOMAIN_COLOR.get(domain, "#fff")
                st.markdown(f"""
<div style="margin-bottom:12px">
  <div style="display:flex;justify-content:space-between">
    <span style="color:{color};font-weight:600;text-transform:capitalize">{domain}</span>
    <span style="color:#9ca3af;font-size:0.8rem">{s['total_queries']} queries</span>
  </div>
  <div style="font-size:0.85rem;color:#6b7280">
    Avg: {round(s['avg_confidence']*100,1)}% &nbsp;|&nbsp;
    Low: {s['low_confidence_count']}
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.caption("No queries yet")

    except Exception:
        st.caption("Stats unavailable")

# ============================================================
# Query input
# ============================================================
with col_query:

    query = st.text_input(
        "Your question:",
        placeholder="e.g. What is insider trading? / How do I use decorators in Python?"
    )

    ask = st.button("Ask", use_container_width=True, type="primary")

    if ask and query.strip():

        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"query": query},
                    timeout=600
                )
                data = resp.json()

            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
                st.stop()

        st.divider()

        # Domain + confidence metrics
        m1, m2, m3 = st.columns(3)

        with m1:
            badge = DOMAIN_BADGE.get(data["domain"], data["domain"])
            st.markdown("**Domain**")
            st.markdown(badge, unsafe_allow_html=True)

        with m2:
            dc = round(data["domain_confidence"] * 100, 1)
            st.metric("Domain Confidence", f"{dc}%")

        with m3:
            ac = round(data["agent_confidence"] * 100, 1)
            st.metric("Answer Confidence", f"{ac}%")

        # Warnings / flags
        if data["low_confidence"]:
            st.warning("⚠️ Low confidence — answer may be unreliable. Try rephrasing.")

        if data.get("retrained"):
            st.success("🔄 Model was retrained on recent low-confidence queries — accuracy improving!")

        # Answer
        # Answer
        st.markdown("#### Answer")

        answer = data["answer"]

        # Better stripping logic (less aggressive, avoids cutting real content)
        for marker in ["Provide a clear legal explanation.", "Answer:", "Solution 0:"]:
            if marker in answer:
                answer = answer.split(marker)[-1].strip()
                break

        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    elif ask and not query.strip():
        st.warning("Please enter a question.")

# ============================================================
# Footer tabs — Agents info + Evaluation
# ============================================================
st.divider()
tab1, tab2 = st.tabs(["🤖 Agents", "📊 Evaluation"])

with tab1:
    try:
        agents_resp = requests.get(f"{API_URL}/agents", timeout=5)
        agents      = agents_resp.json().get("agents", [])

        cols = st.columns(4)
        for i, agent in enumerate(agents):
            color = DOMAIN_COLOR.get(agent["domain"], "#fff")
            with cols[i]:
                st.markdown(f"""
<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px">
  <div style="color:{color};font-weight:700;font-size:1rem;margin-bottom:8px">{agent['name']}</div>
  <div style="color:#6b7280;font-size:0.78rem">Summarizer</div>
  <div style="color:#d1d5db;font-size:0.8rem;margin-bottom:6px">{agent['summarizer'].split('/')[-1]}</div>
  <div style="color:#6b7280;font-size:0.78rem">Reasoner</div>
  <div style="color:#d1d5db;font-size:0.8rem">phi-2</div>
</div>
""", unsafe_allow_html=True)

    except Exception:
        st.caption("Agents info unavailable")

with tab2:
    st.markdown("Run `python evaluation.py` to generate metrics.")

    if os.path.exists("evaluation_results.json"):
        with open("evaluation_results.json") as f:
            eval_data = json.load(f)

        m = eval_data["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Domain Accuracy",    f"{round(m['domain_accuracy']*100,1)}%")
        c2.metric("Avg Confidence",     f"{round(m['avg_confidence']*100,1)}%")
        c3.metric("Hallucination Rate", f"{round(m['hallucination_rate']*100,1)}%")
        c4.metric("Avg Response Time",  f"{round(m['avg_response_time'],1)}s")

    else:
        st.caption("No evaluation results yet.")