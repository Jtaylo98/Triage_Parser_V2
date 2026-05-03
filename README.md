# Clinical Triage Parser

Converts unstructured patient narratives into FHIR-style structured JSON via the Anthropic API.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the server
python app.py
```

Then open **http://localhost:5000** in your browser.

## API

### POST /parse

**Request:**
```json
{
  "narrative": "57-year-old male presents with crushing chest pain..."
}
```

**Response:**
```json
{
  "chief_complaint": "crushing chest pain",
  "symptoms_list": ["chest pain", "diaphoresis", "shortness of breath"],
  "duration": "45 minutes",
  "triage_level": "Emergent",
  "flagged_keywords": ["chest pain", "diaphoresis"]
}
```

### triage_level values
| Value | Meaning |
|-------|---------|
| `Routine` | Non-urgent, can wait |
| `Urgent` | Needs attention within hours |
| `Emergent` | Immediate intervention required |

## curl example

```bash
curl -X POST http://localhost:5000/parse \
  -H "Content-Type: application/json" \
  -d '{"narrative": "Patient presents with chest pain and shortness of breath for 30 minutes."}'
```
