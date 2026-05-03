import os
import json
import re
import uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BACKEND_SECRET = os.environ.get("BACKEND_SECRET", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

URGENCY_MAP = {
    "Routine": "Low",
    "Urgent": "Medium",
    "Emergent": "High",
}

SYSTEM_PROMPT = """You are a clinical data parsing engine. Convert the patient narrative into structured JSON.

Return ONLY a JSON object with these exact keys:
- chief_complaint (string, max 5 words)
- symptoms_list (array of strings)
- duration (string)
- triage_level (string: "Routine", "Urgent", or "Emergent")
- flagged_keywords (array of strings indicating high-risk findings)
- confidence (integer 0-100, your confidence in the parse)
- recommended_action (string, one short sentence)

No prose, no markdown fences. JSON only."""


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found: {text[:200]}")
    return json.loads(match.group(0))


@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": "triage-parser", "status": "ok"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/parse", methods=["POST", "OPTIONS"])
def parse():
    if request.method == "OPTIONS":
        return ("", 204)

    if BACKEND_SECRET:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth.split(" ", 1)[1] != BACKEND_SECRET:
            return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    text = data.get("text") or data.get("narrative") or ""

    if not text.strip():
        return jsonify({"error": "text field is required"}), 400

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        raw = message.content[0].text
        parsed = extract_json(raw)

        triage = parsed.get("triage_level", "Routine")
        suspected_urgency = URGENCY_MAP.get(triage, "Low")

        return jsonify({
            "chief_complaint": parsed.get("chief_complaint", ""),
            "duration": parsed.get("duration", ""),
            "associated_symptoms": parsed.get("symptoms_list", []),
            "suspected_urgency": suspected_urgency,
            "confidence": int(parsed.get("confidence", 75)),
            "recommended_action": parsed.get("recommended_action", ""),
            "flagged_keywords": parsed.get("flagged_keywords", []),
            "pushed_to_openemr": False,
            "encounter_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
