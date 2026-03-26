import torch
import faiss
import pickle

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from shared_models import phi_tokenizer, phi_model, embedding_model, DEVICE


class LegalAgent:

    def __init__(self):

        print("Initializing Legal Agent...")

        # -----------------------------
        # Shared models (loaded once)
        # -----------------------------
        self.phi_tokenizer   = phi_tokenizer
        self.phi_model       = phi_model
        self.embedding_model = embedding_model
        self.device          = DEVICE

        # -----------------------------
        # Load Legal Summarizer
        # -----------------------------
        summarizer_model = "InfurnusWolf/legal_summarizer"

        # IMPORTANT: tokenizer from base LED model
        self.sum_tokenizer = AutoTokenizer.from_pretrained(
            "allenai/led-base-16384"
        )

        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(
            summarizer_model
        ).to(self.device)

        print("Legal summarizer loaded")

        # -----------------------------
        # Load FAISS index
        # -----------------------------
        self.index = faiss.read_index("legal_index.faiss")

        print("FAISS index loaded")

        # -----------------------------
        # Load chunk database
        # -----------------------------
        with open("legal_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print("Loaded", len(self.chunks), "legal chunks")


    def retrieve(self, query, top_k=3):

        query_embedding = self.embedding_model.encode([query])

        distances, indices = self.index.search(query_embedding, top_k)

        return [self.chunks[i] for i in indices[0]]


    def summarize(self, text):

        inputs = self.sum_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        outputs = self.sum_model.generate(**inputs, max_new_tokens=120)

        return self.sum_tokenizer.decode(outputs[0], skip_special_tokens=True)


    def generate_answer(self, query, summaries):

        context = " ".join(summaries)

        prompt = f"""
You are a legal expert in Indian constitutional law.

Question:
{query}

Case-law summaries:
{context}

Provide a clear legal explanation.
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
            temperature=0.7,
            top_p=0.9
        )

        return self.phi_tokenizer.decode(outputs[0], skip_special_tokens=True)


    def run(self, query):

        print("Retrieving legal documents...")
        docs = self.retrieve(query)

        print("Summarizing legal documents...")
        summaries = [self.summarize(doc) for doc in docs[:2]]

        print("Generating final legal answer...")
        answer = self.generate_answer(query, summaries)

        return answer