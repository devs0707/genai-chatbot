# GenAI Chatbot Service

A stateful chatbot service built with FastAPI and LangChain. It handles multi-turn conversations, ambiguous inputs, adversarial content, and factual queries, returning structured JSON responses via a single `/chat` endpoint and a health check at `/healthz`.

## Features

* **Multi-turn State**: Maintains per-session history in-memory.
* **Adversarial & Ethical Guardrails**: Detects gibberish and profanity, responding politely.
* **Contradiction Detection**: Handles simple factual mismatches (e.g., “Is 30 °C freezing?”).
* **Factual Queries**: Answers basic factual questions using an internal mini-knowledge base.
* **Reservation Intent**: Parses fuzzy time expressions (e.g. “this weekend”), manages slot candidates, and asks clarifying questions.
* **Structured Output**: Always returns valid JSON with `intent`, `slots`, `clarification_needed`, and `response`.
* **Health Check**: Exposes `/healthz` for readiness checks.

## Prerequisites

* Python 3.10+
* OpenAI API key (set via `OPENAI_API_KEY` environment variable)
* Docker (optional, if you choose to containerize later)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/devs0707/genai-chatbot
   cd genai-chatbot
   ```

2. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:

   ```bash
   Set API Key in os.environ["OPENAI_API_KEY"] in main.py file
   ```

## Running Locally

Start the service with Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 80
```

* Chat endpoint: `POST http://localhost:80/chat`
* Health check:  `GET  http://localhost:80/healthz`

### Example Usage

```bash
curl -X POST http://localhost:80/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "1234", "message": "Book a table this weekend or maybe Monday morning"}'
```

```json
{
  "intent": "BookReservation",
  "slots": {
    "date_candidates": ["this weekend", "Monday morning"]
  },
  "clarification_needed": true,
  "response": "Do you prefer this weekend or Monday morning?"
}
```

## Project Structure

```
├── main.py            # FastAPI service entrypoint
├── requirements.txt   # Python dependencies
├── README.md          # This file
```

## Notes

* Sessions are stored in-memory; for production, consider using Redis or another persistent store.
* The system prompt in `main.py` (`SYSTEM_TEMPLATE`) defines all conversational logic — feel free to adjust the rules or add new behaviors.

## License

MIT © Your Name or Organization
