import torch
import torch.nn.functional as F
import faiss
import pickle

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from shared_models import phi_tokenizer, phi_model, embedding_model, DEVICE


class FinanceAgent:

    def __init__(self):

        print("Initializing Finance Agent...")

        # -----------------------------
        # Shared models (loaded once)
        # -----------------------------
        self.phi_tokenizer   = phi_tokenizer
        self.phi_model       = phi_model
        self.embedding_model = embedding_model
        self.device          = DEVICE

        # -----------------------------
        # Load Finance Summarizer
        # -----------------------------
        summarizer_model = "InfurnusWolf/finance_summarizer"

        self.sum_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(
            summarizer_model
        ).to(self.device)

        print("Finance summarizer loaded")

        # -----------------------------
        # Load FAISS index
        # -----------------------------
        self.index = faiss.read_index("finance_index.faiss")

        print("FAISS index loaded")

        # -----------------------------
        # Load chunk database
        # -----------------------------
        with open("finance_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print("Loaded", len(self.chunks), "finance chunks")


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

        outputs = self.sum_model.generate(**inputs, max_new_tokens=500)

        summary = self.sum_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # If summarizer produces garbage (repetitive tokens), use raw text instead
        words = summary.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:   # less than 30% unique words = garbage
                return text[:300]    # fall back to first 300 chars of raw chunk

        return summary



    # -----------------------------------------
    # Check if summarizer output is garbage
    # -----------------------------------------
    def is_garbage(self, text):
        if len(text.strip()) < 20:
            return True
        words  = text.strip().split()
        unique = set(words)
        if len(words) > 10 and len(unique) / len(words) < 0.3:
            return True
        return False

    def generate_answer(self, query, summaries):

        context = " ".join(summaries)

        prompt = f"""
You are a financial expert and analyst with deep knowledge of markets,
investments, accounting, and economic concepts.

Question:
{query}

Financial document summaries:
{context}

Provide a clear, accurate financial explanation. Note that this is for
informational purposes and not personal financial advice.
"""

        inputs = self.phi_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        outputs = self.phi_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        full_output = self.phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output.split("Answer:")[-1].strip()
        if not answer:
            answer = full_output[len(self.phi_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
        return answer


    def confidence_score(self, query, docs):

        query_emb = self.embedding_model.encode([query],  convert_to_tensor=True)
        doc_emb   = self.embedding_model.encode(docs[:2], convert_to_tensor=True)

        scores = F.cosine_similarity(query_emb, doc_emb)

        return round(float(scores.mean().item()), 4)


    def run(self, query):

        print("Retrieving finance documents...")
        docs = self.retrieve(query)

        print("Summarizing finance documents...")
        summaries = []
        for doc in docs[:2]:
            summary = self.summarize(doc)
            if self.is_garbage(summary):
                print("  Summarizer output was garbage — using raw chunk")
                summaries.append(doc[:400])
            else:
                summaries.append(summary)

        print("Generating final finance answer...")
        answer = self.generate_answer(query, summaries)

        score = self.confidence_score(query, docs)
        print(f"Confidence score: {score}")

        return {"answer": answer, "confidence": score}