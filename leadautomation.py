"""
Streamlit Lead Qualification & Proposal — Local-Only (No DB, No API Keys)
=======================================================================
- No databases: leads persist to a local JSON file.
- No external APIs: emails saved to disk, notifications to a log, calls simulated as files.
- LLM via local Ollama + Gemma 3 (fallback template if Ollama not running).

Run
---
1) Install deps:
   pip install streamlit pandas langchain langchain-community
2) Ensure Ollama is installed and model pulled:
   ollama pull gemma3
3) Start app:
   streamlit run streamlit_app.py

Folders created on first run
----------------------------
- data/leads.json                 — persisted leads
- outbox/emails/                  — saved "emails" (markdown)
- outbox/call_requests/           — simulated outbound call requests (json)
- outbox/notifications.log        — notifications log
- proposals/                      — generated proposals (markdown)
"""
from __future__ import annotations
import os
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st

# Optional LLM via Ollama
LLM_AVAILABLE = True
try:
    from langchain_community.chat_models import ChatOllama
    from langchain.prompts import ChatPromptTemplate
except Exception:
    LLM_AVAILABLE = False

APP_ROOT = Path(__file__).parent
DATA_DIR = APP_ROOT / "data"
OUTBOX_DIR = APP_ROOT / "outbox"
EMAIL_DIR = OUTBOX_DIR / "emails"
CALL_REQ_DIR = OUTBOX_DIR / "call_requests"
NOTIF_LOG = OUTBOX_DIR / "notifications.log"
PROPOSALS_DIR = APP_ROOT / "proposals"

for p in [DATA_DIR, EMAIL_DIR, CALL_REQ_DIR, PROPOSALS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

LEADS_JSON = DATA_DIR / "leads.json"

DEFAULT_SETTINGS = {
    "OLLAMA_MODEL": "gemma3",
    "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
    "QUAL_THRESHOLD": 10000.0,
}

# -----------------------------
# Utilities
# -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def notify(msg: str):
    line = f"{now_iso()} | {msg}\n"
    with open(NOTIF_LOG, "a", encoding="utf-8") as f:
        f.write(line)
    st.toast(msg)


def load_leads() -> List[Dict[str, Any]]:
    if not LEADS_JSON.exists():
        return []
    try:
        return json.loads(LEADS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_leads(leads: List[Dict[str, Any]]):
    LEADS_JSON.write_text(json.dumps(leads, indent=2), encoding="utf-8")


def upsert_lead(leads: List[Dict[str, Any]], lead: Dict[str, Any]):
    idx = next((i for i, x in enumerate(leads) if (x.get("Email") or "").lower() == (lead.get("Email") or "").lower()), None)
    lead.setdefault("Status", "New")
    lead.setdefault("LastActionAt", now_iso())
    if idx is None:
        leads.append(lead)
    else:
        # keep status if already beyond New/Updated unless we're resyncing
        existing = leads[idx]
        for k, v in lead.items():
            existing[k] = v
        if existing.get("Status") in ("New", "Updated"):
            existing["Status"] = "Updated"
        existing["LastActionAt"] = now_iso()


def as_df(leads: List[Dict[str, Any]]) -> pd.DataFrame:
    if not leads:
        return pd.DataFrame(columns=["Name","Email","Company","UseCase","Budget","Phone","Status","ProposalPath","CallTranscript","LastActionAt","Notes"])
    return pd.DataFrame(leads)


# -----------------------------
# LLM: Proposal Generation
# -----------------------------
PROPOSAL_PROMPT = None
if LLM_AVAILABLE:
    PROPOSAL_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a sales solutions architect who drafts concise, tailored automation proposals."),
        ("human", (
            "Draft a crisp proposal (<= 2 pages) for {company} based on this lead:\n\n"
            "- Prospect: {name}\n- Email: {email}\n- Use case: {use_case}\n- Budget: ${budget}\n\n"
            "Structure:\n1) Problem summary\n2) Proposed automation solution (people, process, tech)\n3) Architecture (bullet points)\n4) Timeline & milestones\n5) Pricing in the stated budget\n6) Next steps (CTA).\n"
        )),
    ])


def generate_proposal_md(lead: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    if LLM_AVAILABLE:
        try:
            llm = ChatOllama(model=cfg.get("OLLAMA_MODEL","gemma3"), base_url=cfg.get("OLLAMA_BASE_URL"), temperature=0.3)
            text = (PROPOSAL_PROMPT | llm).invoke({
                "company": lead.get("Company", "(Company)"),
                "name": lead.get("Name", "(Name)"),
                "email": lead.get("Email", ""),
                "use_case": lead.get("UseCase", lead.get("AutomationNeed", "")),
                "budget": lead.get("Budget", "N/A"),
            }).content
            return text
        except Exception as e:
            notify(f"LLM unavailable, using template: {e}")
    # Fallback template
    return (
        f"# Automation Proposal — {lead.get('Company','(Company)')}\n\n"
        f"**Prospect:** {lead.get('Name','')}  \n"
        f"**Email:** {lead.get('Email','')}  \n"
        f"**Use case:** {lead.get('UseCase', lead.get('AutomationNeed',''))}  \n\n"
        "## 1) Problem summary\nDescribe the current pain points and desired outcomes.\n\n"
        "## 2) Proposed automation solution\n- People: roles and responsibilities\n- Process: key steps and governance\n- Tech: LLM + orchestration + integrations (swappable)\n\n"
        "## 3) Architecture (bullets)\n- Data intake -> Processing -> LLM -> Output\n- Observability & logging\n- Security & access\n\n"
        "## 4) Timeline & milestones\n- Week 1–2: Discovery & design\n- Week 3–4: MVP build\n- Week 5–6: Pilot & iteration\n\n"
        "## 5) Pricing\n- Fixed fee within stated budget with clear deliverables.\n\n"
        "## 6) Next steps\n- Reply to confirm and schedule a working session.\n"
    )


def save_proposal(company: str, content: str) -> str:
    safe = "".join(ch for ch in (company or "proposal") if ch.isalnum() or ch in ("_","-"," "))
    safe = safe.strip().replace(" ", "_") or "proposal"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = PROPOSALS_DIR / f"{safe}_{ts}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


def save_email(to_email: str, subject: str, html: str, attachments: Optional[List[Dict[str,str]]] = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_subj = "".join(ch for ch in subject if ch.isalnum() or ch in (" ","-","_"))[:80].strip().replace(" ","_")
    path = EMAIL_DIR / f"{ts}__{to_email}__{safe_subj}.md"
    body = [f"# To: {to_email}", f"# Subject: {subject}", "", html, ""]
    if attachments:
        body.append("## Attachments")
        for att in attachments:
            body.append(f"- {att['filename']} -> {att['path']}")
    path.write_text("\n".join(body), encoding="utf-8")
    notify(f"Email saved: {path.name} -> {to_email}")
    return str(path)


def trigger_local_call(lead: Dict[str, Any]) -> str:
    payload = {
        "assistantId": "local-assistant",
        "phoneNumber": lead.get("Phone"),
        "customer": {"name": lead.get("Name"), "email": lead.get("Email"), "company": lead.get("Company")},
        "metadata": {"leadEmail": lead.get("Email")},
        "webhookUrl": "(local-streamlit)",
        "synthesis": {"prompt": "Friendly sales agent confirming proposal need."},
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CALL_REQ_DIR / f"call_{lead.get('Email','unknown')}_{ts}.json"
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    notify(f" Simulated call created: {os.path.basename(path)}")
    return str(path)


def auto_qualify(lead: Dict[str, Any], threshold: float) -> str:
    budget = float(lead.get("Budget") or 0)
    qualified = budget >= threshold
    status = "Qualified" if qualified else "Unqualified"
    lead["Status"] = status
    lead["LastActionAt"] = now_iso()
    if qualified:
        trigger_local_call(lead)
    return status


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Lead Automation (Local)", layout="wide")

# Session state
if "settings" not in st.session_state:
    st.session_state.settings = DEFAULT_SETTINGS.copy()
if "leads" not in st.session_state:
    st.session_state.leads = load_leads()

settings = st.session_state.settings
leads = st.session_state.leads

st.title("Lead Qualification & Proposal — Local (No DB, No APIs)")
with st.sidebar:
    st.header("Settings")
    settings["OLLAMA_MODEL"] = st.text_input("Ollama Model", settings.get("OLLAMA_MODEL","gemma3"))
    settings["OLLAMA_BASE_URL"] = st.text_input("Ollama Base URL", settings.get("OLLAMA_BASE_URL","http://127.0.0.1:11434"))
    settings["QUAL_THRESHOLD"] = st.number_input("Qualification Threshold ($)", min_value=0.0, value=float(settings.get("QUAL_THRESHOLD",10000.0)), step=1000.0)
    if st.button("Save Settings"):
        st.success("Settings saved (in memory this session).")

    st.divider()
    st.subheader("Outbox")
    if st.button("Open notifications log"):
        if NOTIF_LOG.exists():
            st.code(NOTIF_LOG.read_text(encoding="utf-8")[-4000:], language="text")
        else:
            st.info("No notifications yet.")

tabs = st.tabs(["Leads", "Import CSV", "Calls", "Webhook Simulator", "Outbox Viewer"]) 

# ----- Leads Tab -----
with tabs[0]:
    st.subheader("Leads")
    with st.expander("Add / Update Lead", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Name")
            company = st.text_input("Company")
            phone = st.text_input("Phone")
        with c2:
            email = st.text_input("Email")
            use_case = st.text_input("Use Case")
        with c3:
            budget = st.number_input("Budget ($)", min_value=0.0, step=1000.0)
        add = st.button("Save Lead")
        if add:
            if not email:
                st.error("Email is required.")
            else:
                upsert_lead(leads, {
                    "Name": name, "Email": email, "Company": company, "UseCase": use_case,
                    "Budget": budget, "Phone": phone, "Status": "New", "LastActionAt": now_iso()
                })
                save_leads(leads)
                st.success("Lead saved.")
                st.rerun()

    df = as_df(leads)
    st.dataframe(df, use_container_width=True)

    st.markdown("### Actions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Auto-qualify All"):
            for lead in leads:
                auto_qualify(lead, settings["QUAL_THRESHOLD"])
            save_leads(leads)
            st.success("Qualification run complete.")
            st.rerun()
    with c2:
        target = st.text_input("Trigger Call for Email")
        if st.button("Trigger Call"):
            found = next((x for x in leads if (x.get("Email") or "").lower() == target.lower()), None)
            if not found:
                st.error("Lead not found.")
            else:
                trigger_local_call(found)
                save_leads(leads)
                st.success("Call request saved to outbox.")
    with c3:
        em = st.text_input("Generate Proposal for Email")
        if st.button("Generate & Save Proposal"):
            lead = next((x for x in leads if (x.get("Email") or "").lower() == em.lower()), None)
            if not lead:
                st.error("Lead not found.")
            else:
                md = generate_proposal_md(lead, settings)
                path = save_proposal(lead.get("Company"), md)
                lead["ProposalPath"] = path
                lead["Status"] = "Proposal Sent"
                save_email(lead.get("Email"), f"{lead.get('Company','Your')} Automation Proposal", "<p>Attached proposal is saved locally.</p>", attachments=[{"filename": os.path.basename(path), "path": path}])
                save_leads(leads)
                st.success(f"Proposal saved: {path}")
    with c4:
        st.write("")

# ----- Import CSV -----
with tabs[1]:
    st.subheader("Import leads from CSV")
    st.caption("Headers: Name, Email, Company, UseCase or AutomationNeed, Budget, Phone")
    uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
    if uploaded is not None:
        content = uploaded.read().decode("utf-8", errors="ignore")
        reader = csv.DictReader(content.splitlines())
        count = 0
        for row in reader:
            norm = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            lead = {
                "Name": norm.get("name") or norm.get("prospect") or "",
                "Email": norm.get("email") or "",
                "Company": norm.get("company"),
                "UseCase": norm.get("usecase") or norm.get("automationneed"),
                "Budget": float(norm.get("budget") or 0),
                "Phone": norm.get("phone") or None,
                "Status": "New",
                "LastActionAt": now_iso(),
            }
            if lead["Email"]:
                upsert_lead(leads, lead)
                count += 1
        save_leads(leads)
        st.success(f"Imported {count} leads.")
        if st.button("Auto-qualify imported leads"):
            for lead in leads:
                if lead.get("Status") in ("New","Updated"):
                    auto_qualify(lead, settings["QUAL_THRESHOLD"])
            save_leads(leads)
            st.success("Qualification complete.")
            st.experimental_rerun()

# ----- Calls -----
with tabs[2]:
    st.subheader("Calls")
    emails = [l.get("Email") for l in leads]
    sel = st.selectbox("Select lead", options=["-"] + emails)
    if sel and sel != "-":
        lead = next(x for x in leads if x.get("Email") == sel)
        st.write({k: lead.get(k) for k in ["Name","Email","Company","UseCase","Budget","Status"]})
        if st.button("Trigger Simulated Call"):
            trigger_local_call(lead)
            st.success("Call request created in outbox.")

# ----- Webhook Simulator -----
with tabs[3]:
    st.subheader("Webhook Simulator (Local)")
    em = st.selectbox("Lead Email", options=["-"] + [l.get("Email") for l in leads])
    event_type = st.selectbox("Event Type", ["call.completed", "call.transcript_finalized", "call.summary", "call.no_answer", "call.unanswered"]) 
    transcript = st.text_area("Transcript", "Great chat. Please send a proposal.")
    if st.button("Send Event"):
        if em == "-":
            st.error("Select a lead.")
        else:
            lead = next((x for x in leads if x.get("Email") == em), None)
            if not lead:
                st.error("Lead not found.")
            else:
                if event_type in ("call.no_answer", "call.unanswered"):
                    lead["Status"] = "No Answer"
                    lead["Notes"] = "No pickup from local outbound"
                    save_email(lead.get("Email"), "Follow-up: Let's schedule a quick call", "<p>Hi, we tried reaching you by phone about your automation project.</p><p>You can reply to this email to coordinate a time.</p><p>— Team</p>")
                    notify("📪 No answer — follow-up saved. #lead-no-pickup")
                else:
                    lead["CallTranscript"] = transcript[:15000]
                    wants = any(kw in transcript.lower() for kw in [
                        "send a proposal","yes proposal","email the proposal","want a proposal","please send proposal","yes"
                    ])
                    if wants:
                        md = generate_proposal_md(lead, settings)
                        path = save_proposal(lead.get("Company"), md)
                        lead["ProposalPath"] = path
                        lead["Status"] = "Proposal Sent"
                        save_email(lead.get("Email"), f"{lead.get('Company','Your')} Automation Proposal", "<p>Hi, attached is your tailored automation proposal. Happy to iterate.</p>", attachments=[{"filename": os.path.basename(path), "path": path}])
                        notify(f" Proposal saved for email to {lead.get('Email')}")
                    else:
                        notify(" Call completed; no proposal requested or not detected.")
                lead["LastActionAt"] = now_iso()
                save_leads(leads)
                st.success("Event processed.")

# ----- Outbox Viewer -----
with tabs[4]:
    st.subheader("Outbox Viewer")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Emails**")
        if EMAIL_DIR.exists():
            files = sorted([p for p in EMAIL_DIR.glob("*.md")], reverse=True)
            sel = st.selectbox("Select email file", ["-"] + [f.name for f in files])
            if sel != "-":
                st.code((EMAIL_DIR/sel).read_text(encoding="utf-8"), language="markdown")
        else:
            st.info("No emails yet.")
    with col2:
        st.markdown("**Call Requests**")
        if CALL_REQ_DIR.exists():
            files = sorted([p for p in CALL_REQ_DIR.glob("*.json")], reverse=True)
            sel2 = st.selectbox("Select call request", ["-"] + [f.name for f in files])
            if sel2 != "-":
                st.code((CALL_REQ_DIR/sel2).read_text(encoding="utf-8"), language="json")
        else:
            st.info("No call requests yet.")

    st.markdown("**Proposals**")
    if PROPOSALS_DIR.exists():
        pfiles = sorted([p for p in PROPOSALS_DIR.glob("*.md")], reverse=True)
        sel3 = st.selectbox("Select proposal", ["-"] + [f.name for f in pfiles])
        if sel3 != "-":
            st.code((PROPOSALS_DIR/sel3).read_text(encoding="utf-8"), language="markdown")
    else:
        st.info("No proposals yet.")
