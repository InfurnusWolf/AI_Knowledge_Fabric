"""
self_learning.py
================
Stores every query + answer + confidence.
Low confidence queries are flagged for retraining.

Flow:
  Every query → saved to feedback_dataset.json
  If confidence < 0.60 → flagged as needs_review = True
  When enough flagged queries accumulate → retrain summarizer

Usage:
  from self_learning import FeedbackLoop
  feedback = FeedbackLoop()
  feedback.store(domain, query, context, answer, confidence)
  feedback.retrain_if_ready(domain)
"""

import os
import json
import torch
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)


# ============================================================
# Config
# ============================================================
FEEDBACK_FILE      = "feedback_dataset.json"
RETRAIN_THRESHOLD  = 50       # retrain after this many low-confidence queries
CONFIDENCE_CUTOFF  = 0.60     # below this = flagged

DOMAIN_SUMMARIZER = {
    "coding":  ("InfurnusWolf/coding_summarizer",  "t5-small",            "./models/coding_summarizer"),
    "medical": ("InfurnusWolf/medical_summarizer", "google/flan-t5-base", "./models/medical_summarizer"),
    "finance": ("InfurnusWolf/finance_summarizer", "google/flan-t5-base", "./models/finance_summarizer"),
    "legal":   ("InfurnusWolf/legal_summarizer",   "allenai/led-base-16384", "./models/legal_summarizer"),
}


class FeedbackLoop:

    def __init__(self):

        # Load existing feedback or start fresh
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                self.data = json.load(f)
        else:
            self.data = []

        print(f"FeedbackLoop loaded — {len(self.data)} entries so far")


    # -----------------------------------------
    # Store every query result
    # Called by manager_agent after every query
    # -----------------------------------------
    def store(self, domain, query, context, answer, confidence):

        entry = {
            "timestamp":    datetime.now().isoformat(),
            "domain":       domain,
            "query":        query,
            "context":      context,
            "answer":       answer,
            "confidence":   confidence,
            "needs_review": confidence < CONFIDENCE_CUTOFF
        }

        self.data.append(entry)

        with open(FEEDBACK_FILE, "w") as f:
            json.dump(self.data, f, indent=2)

        if entry["needs_review"]:
            print(f"  [FeedbackLoop] Low confidence ({confidence}) — flagged for retraining")

        return entry


    # -----------------------------------------
    # Count how many low confidence entries
    # exist for a given domain
    # -----------------------------------------
    def low_confidence_count(self, domain):

        return sum(
            1 for e in self.data
            if e["domain"] == domain and e["needs_review"]
        )


    # -----------------------------------------
    # Get all low confidence entries for domain
    # These become new training data
    # -----------------------------------------
    def get_training_data(self, domain):

        return [
            e for e in self.data
            if e["domain"] == domain and e["needs_review"]
        ]


    # -----------------------------------------
    # Retrain summarizer if enough low
    # confidence queries have accumulated
    # -----------------------------------------
    def retrain_if_ready(self, domain):

        count = self.low_confidence_count(domain)

        print(f"  [FeedbackLoop] {domain}: {count}/{RETRAIN_THRESHOLD} low confidence queries")

        if count < RETRAIN_THRESHOLD:
            return False

        print(f"\n  [FeedbackLoop] Threshold reached — retraining {domain} summarizer...")
        self._retrain(domain)
        return True


    # -----------------------------------------
    # Fine-tune the summarizer on low
    # confidence query-context-answer pairs
    # -----------------------------------------
    def _retrain(self, domain):

        hf_repo, base_tokenizer, output_dir = DOMAIN_SUMMARIZER[domain]

        training_entries = self.get_training_data(domain)

        print(f"  Training on {len(training_entries)} new examples...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
        model     = AutoModelForSeq2SeqLM.from_pretrained(hf_repo)
        model     = model.to(device)

        # Build dataset from low confidence queries
        # input  = query + context (what was retrieved)
        # target = answer (what Phi-2 generated — used as pseudo-label)
        pairs = []
        for e in training_entries:
            pairs.append({
                "input_text":  f"summarize: {e['query']} {e['context']}"[:800],
                "target_text": e["answer"][:200]
            })

        ds = Dataset.from_list(pairs)

        def tokenize(batch):
            model_inputs = tokenizer(
                batch["input_text"],
                max_length=512,
                truncation=True,
                padding="max_length"
            )
            labels = tokenizer(
                batch["target_text"],
                max_length=128,
                truncation=True,
                padding="max_length"
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

        os.makedirs(output_dir, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=4,
            learning_rate=3e-5,
            fp16=(device == "cuda"),
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        )

        trainer.train()

        # Save + upload updated model
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"  Retrained model saved to {output_dir}")

        # Upload to HuggingFace so agents pick it up next restart
        model.push_to_hub(hf_repo)
        tokenizer.push_to_hub(hf_repo)

        print(f"  Uploaded to HuggingFace: {hf_repo}")

        # Mark retrained entries so they aren't used again
        for entry in self.data:
            if entry["domain"] == domain and entry["needs_review"]:
                entry["needs_review"]  = False
                entry["was_retrained"] = True

        with open(FEEDBACK_FILE, "w") as f:
            json.dump(self.data, f, indent=2)

        print(f"  [FeedbackLoop] Retraining complete for {domain}")