# app.py
import streamlit as st
import math
from datetime import datetime
import graphviz
from build_rag_and_decisions import (
    run_pipeline,
    run_text_query_dynamic,
    get_decision_timeline_enhanced
)

# ---- Streamlit Page Config ----
st.set_page_config(page_title="Elyx Decisions", page_icon="💊", layout="wide")
st.title("💊 Elyx Decision Assistant")
st.markdown("A clinical reasoning assistant powered by RAG + Groq LLM")

# ---- Global CSS Override (Force black text everywhere) ----
global_style = """
<style>
html, body, [class*="css"]  {
    color: black !important;
}
.user-bubble {background:#000;color:white;padding:10px;border-radius:10px;margin:5px;max-width:70%;float:right;clear:both;}
.assistant-bubble {background:#ECECEC;color:black;padding:10px;border-radius:10px;margin:5px;max-width:70%;float:left;clear:both;}
.timestamp {font-size:0.7em;color:black;margin-top:2px;}
.role-text {color:black !important;}
</style>
"""
st.markdown(global_style, unsafe_allow_html=True)

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📅 Timeline & Flow", "🚀 Pipeline Explorer"])

# ================= CHAT TAB =================
with tab1:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def render_chat():
        for role, msg, ts in st.session_state.chat_history:
            bubble = "user-bubble" if role == "user" else "assistant-bubble"
            st.markdown(f"""
            <div class="{bubble}">
                {msg}
                <div class="timestamp">{ts}</div>
            </div>
            """, unsafe_allow_html=True)

    render_chat()

    with st.form("chat-form", clear_on_submit=True):
        user_input = st.text_input("Type your decision message...", "")
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        st.session_state.chat_history.append(("user", user_input, datetime.now().strftime("%H:%M")))
        with st.spinner("Thinking..."):
            decision = run_text_query_dynamic(user_input)
            reply = decision["rationale_text"]
        st.session_state.chat_history.append(("assistant", reply, datetime.now().strftime("%H:%M")))
        st.rerun()

# ================= TIMELINE TAB =================
with tab2:
    st.subheader("📅 Decision Timeline & Flow")

    colA, colB, colC = st.columns([1,1,2])
    with colA:
        type_filter = st.multiselect(
            "Filter types",
            options=["diet_change", "supplement_start", "exercise_update", "diagnostic_order", "unknown"],
            default=["diet_change", "supplement_start", "exercise_update", "diagnostic_order", "unknown"]
        )
    with colB:
        success_filter = st.multiselect(
            "Outcome",
            options=["worked", "did_not_work", "neutral", "unclear"],
            default=["worked", "did_not_work", "neutral", "unclear"]
        )
    with colC:
        st.caption("Tip: outcomes are computed from your wearables and biomarker panels around each decision date.")

    timeline = get_decision_timeline_enhanced()
    items = [e for e in timeline if e["type"] in type_filter and e["success"] in success_filter]

    if not items:
        st.info("No timeline items to display with current filters.")
    else:
        items = sorted(items, key=lambda x: x["date"])

        def badge(txt, color):
            return f"<span style='background:{color};color:white;padding:2px 8px;border-radius:999px;font-size:0.8em'>{txt}</span>"

        def delta_chip(label, val, invert=False, decimals=1):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return ""
            up_good = not invert
            arrow = "↑" if val > 0 else ("↓" if val < 0 else "→")
            good = (val > 0 and up_good) or (val < 0 and not up_good)
            bg = "#E6F4EA" if good else "#FDECEA" if val != 0 else "#EEE"
            color = "#137333" if good else "#C5221F" if val != 0 else "#444"
            shown = round(val, decimals)
            return f"<span class='chip' style='background:{bg};color:{color}'>{label} {arrow} {shown}</span>"

        for e in items:
            color = {"worked":"#16a34a","did_not_work":"#dc2626","neutral":"#64748b","unclear":"#a3a3a3"}[e["success"]]
            header = f"**{e['date']}** {badge(e['type'], '#2563eb')} {badge(e['success'], color)}"
            st.markdown(header, unsafe_allow_html=True)

            # Role line with black font
            st.markdown(
                f"<div class='role-text'>- <b>Role:</b> {e['created_by_role']}</div>",
                unsafe_allow_html=True
            )

# ================= PIPELINE TAB =================
with tab3:
    st.subheader("🚀 Pipeline Explorer")
    if st.button("Run Pipeline Now"):
        with st.spinner("Running pipeline..."):
            out = run_pipeline()
        st.success("Pipeline complete.")
        st.json(out)
