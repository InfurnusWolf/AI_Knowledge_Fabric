from legal_agent    import LegalAgent
from coding_agent   import CodingAgent
from medical_agent  import MedicalAgent
from finance_agent  import FinanceAgent
from self_learning  import FeedbackLoop

from transformers import pipeline


class ManagerAgent:

    def __init__(self):

        print("Initializing Manager Agent...")

        # -----------------------------
        # Load all 4 domain agents
        # -----------------------------
        self.legal_agent   = LegalAgent()
        self.coding_agent  = CodingAgent()
        self.medical_agent = MedicalAgent()
        self.finance_agent = FinanceAgent()

        # -----------------------------
        # Load zero-shot classifier
        # -----------------------------
        print("Loading domain classifier...")

        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

        self.candidate_labels = ["coding", "legal", "medical", "finance"]

        self.CONFIDENCE_THRESHOLD = 0.60

        # -----------------------------
        # Self-learning feedback loop
        # -----------------------------
        self.feedback = FeedbackLoop()

        print("Manager Agent ready\n")


    def classify(self, query):

        result = self.classifier(
            query,
            candidate_labels=self.candidate_labels
        )

        domain     = result["labels"][0]
        confidence = round(result["scores"][0], 4)

        return domain, confidence


    def route(self, domain):

        routing_map = {
            "coding":  self.coding_agent,
            "legal":   self.legal_agent,
            "medical": self.medical_agent,
            "finance": self.finance_agent,
        }

        return routing_map[domain]


    def run(self, query):

        print(f"Query: {query}")
        print("-" * 50)

        # Step 1 — classify domain
        print("Classifying domain...")
        domain, domain_confidence = self.classify(query)

        print(f"Domain     : {domain}")
        print(f"Domain confidence : {domain_confidence}")
        print("-" * 50)

        # Step 2 — route to correct agent
        agent = self.route(domain)

        # Step 3 — get answer
        result = agent.run(query)

        if isinstance(result, dict):
            answer           = result["answer"]
            agent_confidence = result["confidence"]
            context          = result.get("context", "")
        else:
            answer           = result
            agent_confidence = domain_confidence
            context          = ""

        low_confidence = agent_confidence < self.CONFIDENCE_THRESHOLD

        if low_confidence:
            print(f"WARNING: Low confidence ({agent_confidence}) — answer may be unreliable")

        # Step 4 — store in feedback loop
        # Every query is saved, low confidence ones
        # are flagged and used for retraining
        self.feedback.store(
            domain=domain,
            query=query,
            context=context,
            answer=answer,
            confidence=agent_confidence
        )

        # Step 5 — check if retraining is needed
        self.feedback.retrain_if_ready(domain)

        return {
            "query":             query,
            "domain":            domain,
            "domain_confidence": domain_confidence,
            "agent_confidence":  agent_confidence,
            "low_confidence":    low_confidence,
            "answer":            answer
        }