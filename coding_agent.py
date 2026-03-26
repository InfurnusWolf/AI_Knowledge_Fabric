import torch
import torch.nn.functional as F
import faiss
import pickle

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from shared_models import phi_tokenizer, phi_model, embedding_model, DEVICE


class CodingAgent:

    def __init__(self):

        print("Initializing Coding Agent...")

        # -----------------------------
        # Shared models (loaded once)
        # -----------------------------
        self.phi_tokenizer   = phi_tokenizer
        self.phi_model       = phi_model
        self.embedding_model = embedding_model
        self.device          = DEVICE

        # -----------------------------
        # Load Coding Summarizer
        # -----------------------------
        summarizer_model = "InfurnusWolf/coding_summarizer"

        self.sum_tokenizer = AutoTokenizer.from_pretrained("t5-small")

        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(
            summarizer_model
        ).to(self.device)

        print("Coding summarizer loaded")

        # -----------------------------
        # Load FAISS index
        # -----------------------------
        self.index = faiss.read_index("coding_index.faiss")

        print("FAISS index loaded")

        # -----------------------------
        # Load chunk database
        # -----------------------------
        with open("coding_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        print("Loaded", len(self.chunks), "coding chunks")


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
You are an expert software engineer and coding assistant.

Question:
{query}

Code documentation summaries:
{context}

Provide a clear explanation with working code examples.
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
            temperature=0.2,
            top_p=0.95
        )

        full_output = self.phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output.split("Provide a clear explanation")[-1].strip()
        if not answer:
            answer = full_output[len(self.phi_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
        return answer


    def confidence_score(self, query, docs):

        query_emb = self.embedding_model.encode([query],  convert_to_tensor=True)
        doc_emb   = self.embedding_model.encode(docs[:2], convert_to_tensor=True)

        scores = F.cosine_similarity(query_emb, doc_emb)

        return round(float(scores.mean().item()), 4)


    def run(self, query):

        print("Retrieving coding documents...")
        docs = self.retrieve(query)

        print("Summarizing code documentation...")
        summaries = []
        for doc in docs[:2]:
            summary = self.summarize(doc)
            if self.is_garbage(summary):
                print("  Summarizer output was garbage — using raw chunk")
                summaries.append(doc[:400])
            else:
                summaries.append(summary)

        print("Generating final coding answer...")
        answer = self.generate_answer(query, summaries)

        score = self.confidence_score(query, docs)
        print(f"Confidence score: {score}")

        return {"answer": answer, "confidence": score}