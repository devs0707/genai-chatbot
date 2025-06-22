import datetime
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Initialize FastAPI app
app = FastAPI(title="GenAI Chatbot Service")
os.environ["OPENAI_API_KEY"] = "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"
# Initialize the LLM and JSON output parser
llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.5)
json_parser = JsonOutputParser()

# In-memory session store: maps session_id to list of message dicts
sessions: dict[str, list[dict[str, str]]] = {}

# System prompt defining the chat behavior
SYSTEM_TEMPLATE = """
You are a stateful chatbot. Manage multi-turn dialogues, ambiguous inputs,
adversarial content, and factual queries. For each message, follow these steps:

1. Ethical & Adversarial Checks:
   - If the input is gibberish or repeated nonsense tokens, respond with:
     {{
       "intent":"None",
       "slots":{{}},
       "clarification_needed":false,
       "response":"I'm sorry, I didn't catch that—could you rephrase?"
     }}
   - If the input contains profanity or slurs, respond with:
     {{
       "intent":"None",
       "slots":{{}},
       "clarification_needed":false,
       "response":"Let's keep our conversation respectful, please."
     }}

2. Contradiction Detection:
   - If the user poses a factual mismatch (e.g., "Is 30 °C freezing?"), respond with:
     {{
       "intent":"None",
       "slots":{{}},
       "clarification_needed":false,
       "response":"No—30 °C is quite warm, not freezing."
     }}

3. Factual Queries:
   - If the user asks a factual question (e.g., "What's the capital of Australia?"), answer directly in the "response" field using your internal knowledge. Preserve any existing intent and slots by carrying them over.

4. Reservation Intent & Fuzzy-Time Parsing:
   - Detect reservation intent when the user indicates booking or reserving.
   - Extract any fuzzy time expressions into slots.date_candidates (a list of phrases).
   - If exactly one date is unambiguous, set slots.date to that value.
   - If slots.date is not yet set, set clarification_needed to true and ask: "Do you prefer <option1>, <option2>, or <option3>?"
   - Slot must consist of date and time. You have to calculate the time and date for the reservation. Today date is: {datetime}
   - Once Slot is set, ask for what will be the party_size?
   - Now if slot is set and number of reservations is set, ask for confirmation.
   - If no new reservation intent or date expressions detected, retain previous intent and slots and set clarification_needed to false.

Always output strictly valid JSON with keys: intent (string), slots (object), clarification_needed (boolean), and response (string). Do not include any additional text or formatting.

Conversation history:
{history}
User: {message}
You have the chat history, if user is asking some of the questions outside the flow then answer the question directly and bring the user back to the flow.
"""

# Build a PromptTemplate for the system + user input
PROMPT = PromptTemplate.from_template(SYSTEM_TEMPLATE)
# Chain that ties prompt + LLM
chain = LLMChain(llm=llm, prompt=PROMPT)

# Pydantic models for request and response
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    intent: str
    slots: dict
    clarification_needed: bool
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Retrieve or initialize conversation history
    history_entries = sessions.get(req.session_id, [])
    # Format history as text for prompt
    history_text = ""
    for entry in history_entries:
        role = entry['role']
        content = entry['content']
        history_text += f"{role.capitalize()}: {content}\n"

    # Invoke the LLM chain
    raw_output = await chain.apredict(history=history_text, message=req.message, datetime = datetime.date.today())

    # Parse the JSON output
    try:
        parsed = json_parser.parse(raw_output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM output as JSON: {e}")

    # Update session history
    sessions.setdefault(req.session_id, []).append({"role": "user", "content": req.message})
    sessions[req.session_id].append({"role": "assistant", "content": raw_output})

    return ChatResponse(**parsed)

@app.get("/healthz")
def health_check():
    # Quick health check: LLM and chain should be initialized
    status = {
        "llm_initialized": True,
        "chain_configured": True
    }
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=80)
