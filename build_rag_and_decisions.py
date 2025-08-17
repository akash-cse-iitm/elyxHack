import os, re, json
import pandas as pd
from dateutil import parser as dtparse
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
# ---------- CONFIG ----------
CONV_PATH = "conversations_raw.txt"
WEARABLE_PATH = "wearable_seed.csv"           # daily granularity
BIOMARKER_PATH = "biomarkers_seed_200.csv"             # months 0,3,6
OUTPUT_DECISIONS = "decision_post_rag.jsonl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = ".chroma_rag"

# Regex patterns
TS_PAT = re.compile(r"^\s*\d+\.\s*\[?(\d{1,2}/\d{1,2}/\d{2,4})\s*,\s*([0-9]{1,2}:[0-9]{2}\s*(?:AM|PM|am|pm)?)\]?\s*(.+?):\s*(.*)$")
TRIGGERS = [
    (r"time[- ]?restricted eating|TRE|10[- ]?hour eating", "diet_change"),
    (r"\bCGM\b|continuous glucose (monitor|sensor)", "diagnostic_order"),
    (r"advanced blood panel|lipid panel|hs-CRP|ApoB", "diagnostic_order"),
    (r"start.*magnesium", "supplement_start"),
    (r"B-Complex|switch.*supplement", "supplement_start"),
    (r"Zone ?5|interval training", "exercise_update"),
    (r"strength program|mobility", "exercise_update"),
    (r"travel protocol|circadian|blue[- ]light glasses", "exercise_update"),
    (r"DEXA|VO2 max|MRI", "diagnostic_order"),
]
ROLE_HINTS = [
    (re.compile(r"doctor|dr\.|elyx medical", re.I), "doctor"),
    (re.compile(r"nutrition", re.I), "nutritionist"),
    (re.compile(r"pt|elyx lifestyle", re.I), "pt"),
    (re.compile(r"concierge", re.I), "concierge"),
    (re.compile(r"rohan", re.I), "member"),
]

# ---------- HELPERS ----------
def guess_role(sender):
    for patt, role in ROLE_HINTS:
        if patt.search(sender):
            return role
    return "unknown"

def parse_conversations(path):
    rows, msg_id = [], 1
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            m = TS_PAT.match(line)
            if not m:
                if rows: rows[-1]["message"] += " " + line
                continue
            date_s, time_s, sender, msg = m.groups()
            dt = dtparse.parse(f"{date_s} {time_s}")
            rows.append({
                "msg_id": f"msg_{msg_id:04d}",
                "date": dt.date().isoformat(),
                "time": dt.time().isoformat(timespec="minutes"),
                "sender": sender.strip(),
                "role": guess_role(sender.lower()),
                "message": msg.strip(),
            })
            msg_id += 1
    return pd.DataFrame(rows)

def extract_decision_candidates(msg_df):
    cands = []
    for _, r in msg_df.iterrows():
        for patt, d_type in TRIGGERS:
            if re.search(patt, r["message"], re.I):
                cands.append({
                    "date": r["date"],
                    "type": d_type,
                    "created_by_role": r["role"],
                    "msg_id": r["msg_id"],
                    "message": r["message"],
                    "sender": r["sender"]
                })
                break
    return pd.DataFrame(cands)

# ---------- RAG ----------
def build_rag_index(msg_df, wear_df, bio_df):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try: client.delete_collection("elyx_rag")
    except: pass
    col = client.create_collection("elyx_rag")
    embedder = SentenceTransformer(EMBED_MODEL)

    docs, ids, metas = [], [], []

    for _, r in msg_df.iterrows():
        doc = f"[{r['date']} {r['time']}] {r['sender']} ({r['role']}): {r['message']}"
        docs.append(doc); ids.append(r["msg_id"])
        metas.append({"type":"message","date":r["date"]})

    for _, r in wear_df.iterrows():
        d = str(r["date"])
        doc = f"Wearable [{d}]: steps={r['Steps']}, sleep={r['Sleep_hours']}h, HRV={r['HRV_ms']}ms, RHR={r['RHR_bpm']} bpm"
        docs.append(doc); ids.append(f"wear_{d}")
        metas.append({"type":"wearable","date":d})

    for _, r in bio_df.iterrows():
        d = str(r["panel_date"])
        doc = f"Biomarker [{d}]: Glucose={r['fasting_glucose']} mg/dL, LDL={r['ldl']}, HDL={r['hdl']}, BP={r['systolic_bp']}/{r['diastolic_bp']} mmHg"
        docs.append(doc); ids.append(f"lab_{d}")
        metas.append({"type":"biomarker","date":d})

    embs = embedder.encode(docs, normalize_embeddings=True, show_progress_bar=True)
    col.add(documents=docs, ids=ids, embeddings=embs, metadatas=metas)
    return col, embedder

def rag_retrieve(col, embedder, query, top_k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    res = col.query(query_embeddings=q_emb, n_results=top_k)
    return list(zip(res["ids"][0], res["documents"][0]))

# ---------- GROQ ----------
def generate_with_groq(prompt):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # free fast model
        messages=[{"role":"user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return resp.choices[0].message.content

def make_rationale(decision, rag_chunks):
    context = "\n".join([f"[{rid}] {doc}" for rid, doc in rag_chunks])
    instr = ("You are a clinical reasoning assistant. "
             "Explain the rationale for this decision in 3–5 sentences. "
             "Use only the provided context. "
             "Cite evidence IDs like [msg_001], [wear_2025-05-20], [lab_2025-06-01].")
    decision_txt = f"Decision: {decision['type']} on {decision['date']}.\nMessage: {decision['message']}"
    prompt = f"CONTEXT:\n{context}\n\n{instr}\n\n{decision_txt}\n\nRATIONALE:"
    return generate_with_groq(prompt)
# ---------- SINGLE DECISION FROM TEXT ----------
def run_text_query(text, date="2025-08-16", sender="Rohan", role="member"):
    """
    Takes a raw message text and returns a decision (if any).
    """
    # Check if this message matches any trigger
    decision_type = None
    for patt, d_type in TRIGGERS:
        if re.search(patt, text, re.I):
            decision_type = d_type
            break
    if not decision_type:
        return None  # no decision found in this text

    # Build minimal context DataFrames
    msg_df = pd.DataFrame([{
        "msg_id": "msg_custom",
        "date": date,
        "time": "09:00",
        "sender": sender,
        "role": role,
        "message": text
    }])

    wear_df = pd.read_csv(WEARABLE_PATH, parse_dates=["date"])
    wear_df["date"] = wear_df["date"].dt.date
    bio_df = pd.read_csv(BIOMARKER_PATH, parse_dates=["panel_date"])
    bio_df["panel_date"] = bio_df["panel_date"].dt.date

    # Build RAG index
    col, embedder = build_rag_index(msg_df, wear_df, bio_df)

    # Get rationale
    rag_chunks = rag_retrieve(col, embedder, text, top_k=5)
    rationale = make_rationale(
        {"type": decision_type, "date": date, "message": text}, rag_chunks
    )
    ev = [{"type": "message", "ref_id": "msg_custom"}] + [{"type": "rag", "ref_id": rid} for rid, _ in rag_chunks]

    decision = {
        "decision_id": "dec_custom_001",
        "date": date,
        "type": decision_type,
        "parameters": {},
        "rationale_text": rationale,
        "source_evidence": ev,
        "created_by_role": role,
        "message": text
    }
    return decision
def run_text_query_dynamic(text, date="2025-08-16", sender="Rohan", role="member"):
    """
    Takes a raw message text and returns a decision (if any).
    Supports dynamic RAG if no regex triggers match.
    """
   
    # 1. Check regex triggers first
    decision_type = None
    for patt, d_type in TRIGGERS:
        if re.search(patt, text, re.I):
            decision_type = d_type
            break

    # 2. If no regex match, use Groq to dynamically infer trigger
    if not decision_type:
        client= Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = f"""
        Extract the most likely event type from the text below.
        Possible types: diet_change, diagnostic_order, supplement_start, exercise_update
        Return only the event type.

        Text: "{text}"
        """
        # response = g.complete(prompt=prompt)
        response = client.chat.completions.create(
    model="llama3-70b-8192",  # Or whatever model you're using
    messages=[
        {"role": "user", "content": prompt}
    ]
)

        decision = response.choices[0].message.content
        dynamic_type = response.choices[0].message.content.strip()
        # Basic validation
        if dynamic_type in ["diet_change", "diagnostic_order", "supplement_start", "exercise_update"]:
            decision_type = dynamic_type
        else:
            decision_type = "unknown"

    # 3. Build minimal context DataFrames
    msg_df = pd.DataFrame([{
        "msg_id": "msg_custom",
        "date": date,
        "time": "09:00",
        "sender": sender,
        "role": role,
        "message": text
    }])

    wear_df = pd.read_csv(WEARABLE_PATH, parse_dates=["date"])
    wear_df["date"] = wear_df["date"].dt.date
    bio_df = pd.read_csv(BIOMARKER_PATH, parse_dates=["panel_date"])
    bio_df["panel_date"] = bio_df["panel_date"].dt.date

    # 4. Build RAG index
    col, embedder = build_rag_index(msg_df, wear_df, bio_df)

    # 5. Retrieve RAG chunks dynamically
    rag_chunks = rag_retrieve(col, embedder, text, top_k=5)

    # 6. Build rationale
    rationale = make_rationale(
        {"type": decision_type, "date": date, "message": text}, rag_chunks
    )

    ev = [{"type": "message", "ref_id": "msg_custom"}] + [{"type": "rag", "ref_id": rid} for rid, _ in rag_chunks]

    # 7. Construct final decision
    decision = {
        "decision_id": "dec_custom_001",
        "date": date,
        "type": decision_type,
        "parameters": {},
        "rationale_text": rationale,
        "source_evidence": ev,
        "created_by_role": role,
        "message": text
    }
    return decision

# ---------- MAIN PIPELINE ----------
def run_pipeline():
    msg_df = parse_conversations(CONV_PATH)
    cand_df = extract_decision_candidates(msg_df)
    if cand_df.empty:
        return []

    wear_df = pd.read_csv(WEARABLE_PATH, parse_dates=["date"])
    wear_df["date"] = wear_df["date"].dt.date
    bio_df = pd.read_csv(BIOMARKER_PATH, parse_dates=["panel_date"])
    bio_df["panel_date"] = bio_df["panel_date"].dt.date

    col, embedder = build_rag_index(msg_df, wear_df, bio_df)

    decisions = []
    for _, c in cand_df.iterrows():
        rag_chunks = rag_retrieve(col, embedder, c["message"], top_k=5)
        rationale = make_rationale(c, rag_chunks)
        ev = [{"type":"message","ref_id":c["msg_id"]}] + [{"type":"rag","ref_id":rid} for rid,_ in rag_chunks]

        decisions.append({
            "decision_id": f"dec_{len(decisions)+1:03d}",
            "date": c["date"],
            "type": c["type"],
            "parameters": {},
            "rationale_text": rationale,
            "source_evidence": ev,
            "created_by_role": c["created_by_role"],
            "message": c["message"]
        })
    return decisions
def get_decision_timeline():
    """
    Load all decisions generated from pipeline and return them
    in a timeline-friendly format for Streamlit.
    """
    import json
    
    timeline = []
    if not os.path.exists(OUTPUT_DECISIONS):
        return timeline
    
    with open(OUTPUT_DECISIONS, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            timeline.append({
                "date": d["date"],
                "type": d["type"],
                "message": d["message"],
                "rationale": d["rationale_text"],
                "role": d["created_by_role"],
                "success": "worked" if "improve" in d["rationale_text"].lower() else "unclear"
            })
    return timeline
    # ======== OUTCOME EVALUATION & TIMELINE EXPORT ========

from datetime import datetime, timedelta

# Helper: nearest biomarker panel before/after a given date
def _nearest_biomarker_panels(bio_df, anchor_date):
    """Return (prev_row, next_row) biomarker panels around anchor_date (date)."""
    prev = bio_df[bio_df["panel_date"] <= anchor_date].sort_values("panel_date").tail(1)
    nxt  = bio_df[bio_df["panel_date"] >  anchor_date].sort_values("panel_date").head(1)
    prev_row = prev.iloc[0].to_dict() if len(prev) else None
    next_row = nxt.iloc[0].to_dict() if len(nxt) else None
    return prev_row, next_row

# Helper: wearable aggregates in windows
def _wearable_window_agg(wear_df, start_date, end_date):
    mask = (wear_df["date"] >= start_date) & (wear_df["date"] <= end_date)
    w = wear_df.loc[mask]
    if w.empty:
        return {}
    return {
        "Steps_mean": float(w["Steps"].mean()) if "Steps" in w else None,
        "RHR_bpm_mean": float(w["RHR_bpm"].mean()) if "RHR_bpm" in w else None,
        "HRV_ms_mean": float(w["HRV_ms"].mean()) if "HRV_ms" in w else None,
        "Sleep_hours_mean": float(w["Sleep_hours"].mean()) if "Sleep_hours" in w else None,
        "VO2_max_mean": float(w["VO2_max"].mean()) if "VO2_max" in w else None,
        "Stress_index_mean": float(w["Stress_index"].mean()) if "Stress_index" in w else None,
    }

def evaluate_decision_outcome(decision, wear_df, bio_df,
                              pre_days=14, post_days=28):
    """
    Data-driven outcome evaluation based on decision type.

    Rules of thumb:
      - diet_change: expect ↓ fasting_glucose, ↓ triglycerides, ↑ HDL; wearables: ↑ Steps, ↑ HRV, ↓ RHR, ↓ Stress
      - supplement_start: flexible; accept any of (↑ HRV, ↓ RHR, ↑ Sleep, ↓ BP, lipid improvements)
      - exercise_update: expect ↑ Steps/VO2/HRV and ↓ RHR/Stress
      - diagnostic_order: outcome = 'neutral' (info gathering)

    Returns dict with success label, rationale_notes, and metric deltas.
    """
    # Parse dates
    d_date = datetime.fromisoformat(decision["date"]).date()
    pre_start  = d_date - timedelta(days=pre_days)
    pre_end    = d_date - timedelta(days=1)
    post_start = d_date + timedelta(days=1)
    post_end   = d_date + timedelta(days=post_days)

    # Wearable windows
    pre_w  = _wearable_window_agg(wear_df, pre_start, pre_end)
    post_w = _wearable_window_agg(wear_df, post_start, post_end)

    # Biomarker before/after panels
    prev_bio, next_bio = _nearest_biomarker_panels(bio_df, d_date)

    # Compute deltas (post - pre) for wearables
    def _delta(a, b):
        if a is None or b is None:
            return None
        return b - a

    wear_deltas = {}
    for k in ["Steps_mean", "RHR_bpm_mean", "HRV_ms_mean", "Sleep_hours_mean", "VO2_max_mean", "Stress_index_mean"]:
        wear_deltas[k] = _delta(pre_w.get(k), post_w.get(k))

    # Compute biomarker deltas (next - prev) when possible
    bio_deltas = {}
    if prev_bio and next_bio:
        for k in ["fasting_glucose", "ldl", "hdl", "triglycerides", "hs_crp", "vit_d", "systolic_bp", "diastolic_bp"]:
            if (k in prev_bio) and (k in next_bio):
                try:
                    bio_deltas[k] = float(next_bio[k]) - float(prev_bio[k])
                except Exception:
                    bio_deltas[k] = None
    else:
        # fallbacks: single panel available implies no delta
        for k in ["fasting_glucose", "ldl", "hdl", "triglycerides", "hs_crp", "vit_d", "systolic_bp", "diastolic_bp"]:
            bio_deltas[k] = None

    # Threshold rules
    worked_flags = []
    notes = []

    dtype = decision.get("type", "")

    if dtype == "diagnostic_order":
        return {
            "success": "neutral",
            "notes": "Diagnostic order; not outcome-bearing.",
            "wear_pre": pre_w, "wear_post": post_w, "wear_delta": wear_deltas,
            "bio_prev": prev_bio, "bio_next": next_bio, "bio_delta": bio_deltas
        }

    def _add_flag(cond, msg):
        if cond is None:
            return
        worked_flags.append(bool(cond))
        if cond:
            notes.append(f"✓ {msg}")
        else:
            notes.append(f"✗ {msg}")

    # Wearable expectations
    steps_up     = (wear_deltas["Steps_mean"] is not None and wear_deltas["Steps_mean"] > 200)   # +200 steps on avg
    rhr_down     = (wear_deltas["RHR_bpm_mean"] is not None and wear_deltas["RHR_bpm_mean"] < -1.0)
    hrv_up       = (wear_deltas["HRV_ms_mean"] is not None and wear_deltas["HRV_ms_mean"] > 1.0)
    sleep_up     = (wear_deltas["Sleep_hours_mean"] is not None and wear_deltas["Sleep_hours_mean"] > 0.1)
    vo2_up       = (wear_deltas["VO2_max_mean"] is not None and wear_deltas["VO2_max_mean"] > 0.3)
    stress_down  = (wear_deltas["Stress_index_mean"] is not None and wear_deltas["Stress_index_mean"] < -1.0)

    # Biomarker expectations
    g_down = (bio_deltas["fasting_glucose"] is not None and bio_deltas["fasting_glucose"] < -2.0)
    tg_down = (bio_deltas["triglycerides"] is not None and bio_deltas["triglycerides"] < -10.0)
    hdl_up_b = (bio_deltas["hdl"] is not None and bio_deltas["hdl"] > 1.0)
    ldl_down = (bio_deltas["ldl"] is not None and bio_deltas["ldl"] < -5.0)
    hscrp_down = (bio_deltas["hs_crp"] is not None and bio_deltas["hs_crp"] < -0.2)
    bp_down = (
        (bio_deltas["systolic_bp"] is not None and bio_deltas["systolic_bp"] < -3) or
        (bio_deltas["diastolic_bp"] is not None and bio_deltas["diastolic_bp"] < -2)
    )

    if dtype == "diet_change":
        _add_flag(g_down, "Fasting glucose fell")
        _add_flag(tg_down, "Triglycerides fell")
        _add_flag(hdl_up_b, "HDL increased")
        _add_flag(steps_up, "Steps increased")
        _add_flag(hrv_up, "HRV improved")
        _add_flag(rhr_down, "RHR decreased")
        _add_flag(stress_down, "Stress decreased")

    elif dtype == "supplement_start":
        # More flexible: any meaningful improvement counts
        any_improve = any([
            hrv_up, rhr_down, sleep_up, vo2_up, stress_down, bp_down, ldl_down, hdl_up_b, tg_down, g_down, hscrp_down
        ])
        _add_flag(any_improve, "At least one target metric improved")
        # Keep individual notes too
        _add_flag(bp_down, "Blood pressure improved")
        _add_flag(ldl_down, "LDL fell")
        _add_flag(hdl_up_b, "HDL rose")
        _add_flag(hrv_up, "HRV improved")
        _add_flag(rhr_down, "RHR decreased")
        _add_flag(stress_down, "Stress decreased")

    elif dtype == "exercise_update":
        _add_flag(steps_up, "Steps increased")
        _add_flag(vo2_up, "VO2max increased")
        _add_flag(hrv_up, "HRV improved")
        _add_flag(rhr_down, "RHR decreased")
        _add_flag(stress_down, "Stress decreased")

    # Final label
    if not worked_flags:
        label = "unclear"
    else:
        score = sum(1 for f in worked_flags if f)
        label = "worked" if score >= max(1, len(worked_flags)//2) else "did_not_work"

    return {
        "success": label,
        "notes": "; ".join(notes) if notes else "Insufficient data to judge.",
        "wear_pre": pre_w, "wear_post": post_w, "wear_delta": wear_deltas,
        "bio_prev": prev_bio, "bio_next": next_bio, "bio_delta": bio_deltas
    }

def get_decision_timeline_enhanced():
    """
    Returns a list of decisions with outcome evaluation and deltas,
    ready for timeline/flowchart UI.
    """
    import json

    # Load decisions
    decisions = []
    if os.path.exists(OUTPUT_DECISIONS):
        with open(OUTPUT_DECISIONS, "r", encoding="utf-8") as f:
            for line in f:
                decisions.append(json.loads(line))

    # Load data sources
    wear_df = pd.read_csv(WEARABLE_PATH, parse_dates=["date"])
    wear_df["date"] = wear_df["date"].dt.date
    bio_df = pd.read_csv(BIOMARKER_PATH, parse_dates=["panel_date"])
    bio_df["panel_date"] = bio_df["panel_date"].dt.date

    enriched = []
    for d in decisions:
        outcome = evaluate_decision_outcome(d, wear_df, bio_df)
        enriched.append({
            "decision_id": d.get("decision_id"),
            "date": d.get("date"),
            "type": d.get("type"),
            "created_by_role": d.get("created_by_role"),
            "message": d.get("message"),
            "rationale_text": d.get("rationale_text"),
            "success": outcome["success"],
            "outcome_notes": outcome["notes"],
            "wear_delta": outcome["wear_delta"],
            "bio_delta": outcome["bio_delta"],
        })
    return enriched
# ---------- CLI ENTRY ----------
if __name__ == "__main__":
    results = run_pipeline()
    with open(OUTPUT_DECISIONS, "w", encoding="utf-8") as out:
        for d in results:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(results)} grounded decisions to {OUTPUT_DECISIONS}")