# build_decisions_with_rag_groq.py
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
WEARABLE_PATH = "wearable_seed.csv"           
BIOMARKER_PATH = "biomarkers_seed_200.csv"            
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
            if not line: 
                continue
            m = TS_PAT.match(line)
            if not m:
                if rows: 
                    rows[-1]["message"] += " " + line
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
    try: 
        client.delete_collection("elyx_rag")
    except: 
        pass
    col = client.create_collection("elyx_rag")
    embedder = SentenceTransformer(EMBED_MODEL)

    docs, ids, metas = [], [], []

    # messages
    for _, r in msg_df.iterrows():
        doc = f"[{r['date']} {r['time']}] {r['sender']} ({r['role']}): {r['message']}"
        docs.append(doc)
        ids.append(r["msg_id"])
        metas.append({"type":"message","date":r["date"]})

    # wearables (ensure unique IDs)
    for i, r in wear_df.reset_index().iterrows():
        d = str(r["date"])
        doc = f"Wearable [{d}]: steps={r['Steps']}, sleep={r['Sleep_hours']}h, HRV={r['HRV_ms']}ms, RHR={r['RHR_bpm']} bpm"
        docs.append(doc)
        ids.append(f"wear_{d}_{i:04d}")
        metas.append({"type":"wearable","date":d})

    # biomarkers (ensure unique IDs)
    for i, r in bio_df.reset_index().iterrows():
        d = str(r["panel_date"])
        doc = f"Biomarker [{d}]: Glucose={r['fasting_glucose']} mg/dL, LDL={r['ldl']}, HDL={r['hdl']}, BP={r['systolic_bp']}/{r['diastolic_bp']} mmHg"
        docs.append(doc)
        ids.append(f"lab_{d}_{i:04d}")
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
        model="llama-3.1-8b-instant",
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
             "Cite evidence IDs like [msg_001], [wear_2025-05-20_0001], [lab_2025-06-01_0002].")
    decision_txt = f"Decision: {decision['type']} on {decision['date']}.\nMessage: {decision['message']}"
    prompt = f"CONTEXT:\n{context}\n\n{instr}\n\n{decision_txt}\n\nRATIONALE:"
    return generate_with_groq(prompt)

# ---------- PIPELINE ----------
def run_pipeline():
    msg_df = parse_conversations(CONV_PATH)
    cand_df = extract_decision_candidates(msg_df)
    if cand_df.empty:
        return []

    wear_df = pd.read_csv(WEARABLE_PATH)
    wear_df["date"] = pd.to_datetime(wear_df["date"], errors="coerce").dt.date

    bio_df = pd.read_csv(BIOMARKER_PATH)
    bio_df["panel_date"] = pd.to_datetime(bio_df["panel_date"], errors="coerce").dt.date

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

# ---------- MAIN ----------
def main():
    results = run_pipeline()
    with open(OUTPUT_DECISIONS, "w", encoding="utf-8") as out:
        for d in results: 
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(results)} grounded decisions to {OUTPUT_DECISIONS}")
# ---------- CLI ENTRY ----------

  
if __name__ == "__main__":
    results = run_pipeline()
    with open(OUTPUT_DECISIONS, "w", encoding="utf-8") as out:
        for d in results:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(results)} grounded decisions to {OUTPUT_DECISIONS}")