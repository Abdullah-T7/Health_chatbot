from flask import Flask, render_template, request, jsonify, session
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "default_secret")


client = InferenceClient(
  model="meta-llama/Llama-3.1-8B-Instruct",
  token=os.environ["HF_TOKEN"]
)
SYSTEM_PROMPT = """You are a friendly health assistant.
Answer general health questions simply and clearly.
Never diagnose. Always suggest seeing a doctor.
Keep answers to 3-5 sentences."""


BLOCKED = ["overdose", "kill myself", "suicide", "self harm"]
SAFE_MSG = "I can't help with that. Please contact a helpline or doctor."

def is_unsafe(text):
  return any(w in text.lower() for w in BLOCKED)


# ── Routes ───────────────────────────────────────────────

@app.route("/")

def home():
  session["history"] = [{"role": "system", "content": SYSTEM_PROMPT}]
  return render_template("index.html")


@app.route("/chat", methods=["POST"])

def chat():
  
  user_msg = request.json.get("message", "").strip()

  if not user_msg:
    return jsonify({"reply": ""})
  
  if is_unsafe(user_msg):
    return jsonify({"reply": SAFE_MSG})
  
  history = session.get("history", [])
  history.append({"role": "user", "content": user_msg})

  try:
    resp = client.chat_completion(messages=history, max_tokens=300)
    reply = resp.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    session["history"] = history
    return jsonify({"reply": reply})
  
  except Exception as e:
    return jsonify({"reply": f"Error: {e}"})
  
if __name__ == "__main__":
  app.run(debug=True)
