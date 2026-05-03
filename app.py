"""
Clinical Triage Parser — Flask API
Converts unstructured patient narratives into FHIR-style structured JSON.

Usage:
  1. pip install flask anthropic
  2. export ANTHROPIC_API_KEY=sk-ant-...
  3. python app.py
  4. POST to http://localhost:5000/parse  (or open http://localhost:5000 in browser)
"""

import os
import json
from flask import Flask, request, jsonify, render_template_string
import anthropic

app = Flask(__name__)

SYSTEM_PROMPT = """You are a clinical data parsing engine designed to convert unstructured patient narratives into structured, FHIR-compliant JSON payloads for EMR integration.

Read the patient input and return ONLY a JSON object with these exact keys:
- "chief_complaint": string, max 5 words summarizing the main complaint
- "symptoms_list": array of strings, each a distinct symptom mentioned
- "duration": string describing how long symptoms have been present
- "triage_level": exactly one of "Routine", "Urgent", or "Emergent"
- "flagged_keywords": array of high-risk terms present in the narrative (e.g. "chest pain", "stiff neck", "worst headache of my life", "difficulty breathing", "altered consciousness", "cyanosis", "diaphoresis", "syncope", "stroke symptoms", "unresponsive")

Return ONLY the JSON object. No markdown, no code fences, no commentary, no explanation."""

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clinical Triage Parser</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f4; color: #1c1917; min-height: 100vh; padding: 2rem; }
  .container { max-width: 900px; margin: 0 auto; }
  header { margin-bottom: 2rem; }
  .badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px; background: #fee2e2; color: #991b1b; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px; }
  h1 { font-size: 24px; font-weight: 600; margin-bottom: 4px; }
  .subtitle { font-size: 14px; color: #78716c; }
  .panel { background: white; border: 1px solid #e7e5e4; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.25rem; }
  .examples { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }
  .examples span { font-size: 12px; color: #78716c; }
  .ex-btn { font-size: 12px; padding: 4px 12px; border-radius: 20px; border: 1px solid #d6d3d1; background: #fafaf9; color: #57534e; cursor: pointer; }
  .ex-btn:hover { background: #f5f5f4; }
  textarea { width: 100%; min-height: 130px; resize: vertical; font-size: 14px; line-height: 1.6; padding: 12px; border: 1px solid #d6d3d1; border-radius: 8px; font-family: inherit; color: #1c1917; background: #fafaf9; }
  textarea:focus { outline: none; border-color: #a8a29e; box-shadow: 0 0 0 3px rgba(0,0,0,0.05); }
  .actions { display: flex; justify-content: flex-end; margin-top: 12px; }
  button.primary { padding: 9px 22px; border-radius: 8px; border: 1px solid #d6d3d1; background: white; color: #1c1917; font-size: 14px; font-weight: 500; cursor: pointer; }
  button.primary:hover { background: #f5f5f4; }
  button.primary:disabled { opacity: 0.45; cursor: not-allowed; }
  .triage-banner { border-radius: 8px; padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
  .triage-Routine  { background: #dcfce7; color: #166534; }
  .triage-Urgent   { background: #fef9c3; color: #854d0e; }
  .triage-Emergent { background: #fee2e2; color: #991b1b; }
  .triage-label { font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
  .triage-val { font-size: 20px; font-weight: 600; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  .card { background: white; border: 1px solid #e7e5e4; border-radius: 10px; padding: 14px 16px; }
  .card-label { font-size: 11px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #78716c; margin-bottom: 6px; }
  .card-val { font-size: 14px; color: #1c1917; line-height: 1.5; }
  .pill { display: inline-block; font-size: 12px; padding: 3px 10px; border-radius: 20px; margin: 2px 4px 2px 0; border: 1px solid #e7e5e4; background: #fafaf9; }
  .flag-pill { background: #fee2e2; border-color: transparent; color: #991b1b; font-weight: 500; }
  .json-wrap { background: #fafaf9; border: 1px solid #e7e5e4; border-radius: 8px; padding: 14px 16px; }
  .json-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .json-title { font-size: 11px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #78716c; }
  .copy-btn { font-size: 12px; padding: 3px 10px; border-radius: 4px; border: 1px solid #e7e5e4; background: transparent; color: #78716c; cursor: pointer; }
  pre { font-family: "SF Mono", "Fira Code", monospace; font-size: 12px; color: #1c1917; white-space: pre-wrap; word-break: break-all; line-height: 1.6; }
  #output { display: none; }
  #output.visible { display: block; }
  .error-box { background: #fee2e2; border: 1px solid #fca5a5; border-radius: 8px; padding: 12px 16px; display: none; }
  .error-title { font-size: 12px; font-weight: 600; color: #991b1b; margin-bottom: 4px; text-transform: uppercase; }
  .error-body { font-family: monospace; font-size: 12px; color: #991b1b; white-space: pre-wrap; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #d6d3d1; border-top-color: #44403c; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (max-width: 600px) { .grid2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="badge">Clinical</div>
    <h1>Triage Parser</h1>
    <p class="subtitle">Converts unstructured patient narratives into FHIR-ready structured JSON</p>
  </header>

  <div class="panel">
    <div class="examples">
      <span>Examples:</span>
      <button class="ex-btn" onclick="loadExample('routine')">Routine</button>
      <button class="ex-btn" onclick="loadExample('urgent')">Urgent</button>
      <button class="ex-btn" onclick="loadExample('emergent')">Emergent</button>
    </div>
    <textarea id="narrative" placeholder="57-year-old male presents with sudden onset crushing chest pain radiating to the left arm..."></textarea>
    <div class="actions">
      <button class="primary" id="parseBtn" onclick="runParse()">Parse narrative</button>
    </div>
  </div>

  <div class="error-box" id="errorBox">
    <div class="error-title">Error</div>
    <div class="error-body" id="errorBody"></div>
  </div>

  <div id="output">
    <div class="triage-banner" id="triageBanner">
      <span class="triage-label">Triage level</span>
      <span class="triage-val" id="triageVal">—</span>
    </div>
    <div class="grid2">
      <div class="card"><div class="card-label">Chief complaint</div><div class="card-val" id="chiefComplaint">—</div></div>
      <div class="card"><div class="card-label">Duration</div><div class="card-val" id="duration">—</div></div>
    </div>
    <div class="card" style="margin-bottom:12px"><div class="card-label">Symptoms</div><div id="symptomsList" class="card-val">—</div></div>
    <div class="card" style="margin-bottom:12px"><div class="card-label">Flagged keywords</div><div id="flaggedList" class="card-val">—</div></div>
    <div class="json-wrap">
      <div class="json-header">
        <span class="json-title">FHIR JSON payload</span>
        <button class="copy-btn" onclick="copyJson()">Copy</button>
      </div>
      <pre id="jsonOut"></pre>
    </div>
  </div>
</div>

<script>
const EXAMPLES = {
  routine: "Patient is a 34-year-old female with a 5-day history of mild sore throat, low-grade fever of 99.2F, and nasal congestion. No difficulty swallowing. She denies any rash, neck stiffness, or difficulty breathing.",
  urgent: "28-year-old male with known asthma presents with worsening shortness of breath over the past 6 hours. Using rescue inhaler every 2 hours with minimal relief. Moderate wheezing on exam. Peak flow at 55% of personal best.",
  emergent: "72-year-old female brought in by EMS with sudden onset severe headache described as the worst headache of my life, onset 30 minutes ago. Associated with neck stiffness, photophobia, and vomiting. GCS 13. BP 185/110."
};
let lastJson = null;

function loadExample(t) {
  document.getElementById('narrative').value = EXAMPLES[t];
  document.getElementById('output').classList.remove('visible');
  document.getElementById('errorBox').style.display = 'none';
}

async function runParse() {
  const text = document.getElementById('narrative').value.trim();
  if (!text) return;
  const btn = document.getElementById('parseBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Parsing...';
  document.getElementById('errorBox').style.display = 'none';
  document.getElementById('output').classList.remove('visible');

  try {
    const res = await fetch('/parse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ narrative: text })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || JSON.stringify(data));
    lastJson = data;
    renderOutput(data);
  } catch(err) {
    document.getElementById('errorBox').style.display = 'block';
    document.getElementById('errorBody').textContent = err.message;
  }

  btn.disabled = false;
  btn.textContent = 'Parse narrative';
}

function renderOutput(d) {
  const triage = (d.triage_level || 'Unknown').trim();
  document.getElementById('triageBanner').className = 'triage-banner triage-' + triage;
  document.getElementById('triageVal').textContent = triage;
  document.getElementById('chiefComplaint').textContent = d.chief_complaint || '—';
  document.getElementById('duration').textContent = d.duration || '—';
  const s = d.symptoms_list || [];
  document.getElementById('symptomsList').innerHTML = s.length ? s.map(x => `<span class="pill">${x}</span>`).join('') : '<span style="color:#a8a29e">None identified</span>';
  const f = d.flagged_keywords || [];
  document.getElementById('flaggedList').innerHTML = f.length ? f.map(x => `<span class="pill flag-pill">${x}</span>`).join('') : '<span style="color:#a8a29e">None</span>';
  document.getElementById('jsonOut').textContent = JSON.stringify(d, null, 2);
  document.getElementById('output').classList.add('visible');
}

function copyJson() {
  if (!lastJson) return;
  navigator.clipboard.writeText(JSON.stringify(lastJson, null, 2));
}
</script>
</body>
</html>"""


def extract_json(text: str) -> dict:
    """Robustly extract JSON from model output, stripping any markdown fences."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response. Raw output:\n{cleaned[:500]}")
    return json.loads(cleaned[start:end + 1])


@app.route("/")
def index():
    return render_template_string(HTML_UI)


@app.route("/parse", methods=["POST"])
def parse_narrative():
    data = request.get_json(force=True)
    narrative = (data.get("narrative") or "").strip()
    if not narrative:
        return jsonify({"error": "narrative field is required"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY environment variable not set"}), 500

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": narrative}]
    )

    raw = "".join(block.text for block in message.content if hasattr(block, "text"))
    parsed = extract_json(raw)
    return jsonify(parsed)


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠  ANTHROPIC_API_KEY not set.")
        print("   Run: export ANTHROPIC_API_KEY=sk-ant-...\n")
    print("Starting Clinical Triage Parser on http://localhost:5000")
    app.run(debug=True, port=5000)
