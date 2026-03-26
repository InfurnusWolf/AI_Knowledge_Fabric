import torch
import torch.nn.functional as F
import faiss
import pickle

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from shared_models import phi_tokenizer, phi_model, embedding_model, DEVICE


class MedicalAgent:

    def __init__(self):

        print("Initializing Medical Agent...")

        self.phi_tokenizer   = phi_tokenizer
        self.phi_model       = phi_model
        self.embedding_model = embedding_model
        self.device          = DEVICE

        summarizer_model = "InfurnusWolf/medical_summarizer"

        self.sum_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(
            summarizer_model
        ).to(self.device)

        print("Medical summarizer loaded")

        self.index = faiss.read_index("medical_index.faiss")
        print("FAISS index loaded")

        with open("medical_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print("Loaded", len(self.chunks), "medical chunks")


    def retrieve(self, query, top_k=3):

        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.chunks[i] for i in indices[0]]


    def summarize(self, text):

        inputs = self.sum_tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.sum_model.generate(**inputs, max_new_tokens=120)

        summary = self.sum_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return summary


    # -----------------------------------------
    # Check if summarizer output is garbage
    # Detects repetition loops like
    # "Gö Gö Gö" or "weil weil weil"
    # -----------------------------------------
    def is_garbage(self, text):

        if len(text.strip()) < 20:
            return True

        words  = text.strip().split()
        unique = set(words)

        # If more than 50% of words are the same → garbage
        if len(words) > 10 and len(unique) / len(words) < 0.3:
            return True

        return False


    def generate_answer(self, query, context):

        prompt = f"""
You are a medical expert trained on clinical literature and PubMed research.

Question:
{query}

Relevant medical context:
{context}

Provide a clear, accurate medical explanation. Always recommend consulting
a licensed healthcare professional for personal medical decisions.
"""

        inputs = self.phi_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        outputs = self.phi_model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.5,
            top_p=0.9
        )

        return self.phi_tokenizer.decode(outputs[0], skip_special_tokens=True)


    def confidence_score(self, query, docs):

        query_emb = self.embedding_model.encode([query],  convert_to_tensor=True)
        doc_emb   = self.embedding_model.encode(docs[:2], convert_to_tensor=True)
        scores    = F.cosine_similarity(query_emb, doc_emb)
        return round(float(scores.mean().item()), 4)


    def run(self, query):

        print("Retrieving medical documents...")
        docs = self.retrieve(query)

        print("Summarizing medical documents...")
        context_parts = []

        for doc in docs[:2]:
            summary = self.summarize(doc)

            if self.is_garbage(summary):
                # Summarizer produced garbage — use raw doc chunk instead
                print("  Summarizer output was garbage — using raw chunk")
                context_parts.append(doc[:400])
            else:
                context_parts.append(summary)

        context = " ".join(context_parts)

        print("Generating final medical answer...")
        answer = self.generate_answer(query, context)

        score = self.confidence_score(query, docs)
        print(f"Confidence score: {score}")

        return {"answer": answer, "confidence": score}