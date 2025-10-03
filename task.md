### Retrieval + Guardrails – lean FastAPI

**Overview**
Prototype a **minimal retrieval-augmented answering service**.

**Your Task**

- Build a **small text corpus** (10–15 snippets you write).
- A **FastAPI endpoint** `POST /answer` → query in, top-k snippets + naive “answer” out.
- Add **one guardrail** (denylist, budget rule, or other) and explain why it’s practical.
- Compare **two index configs** (e.g., cosine vs dot-product, or k=3 vs k=5) and justify your choice.
- Suggest **2 lightweight monitoring metrics** (e.g., latency, hit-rate/drift) and sketch how you’d track them.

Keep it scrappy but production-minded.