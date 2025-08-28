📈 Lead Qualification & Proposal Automation (Local, No DB, No APIs)

A Streamlit-based demo system for automating the lead → qualification → call → proposal workflow without any external APIs or credentials.

This project shows how sales automation workflows can be prototyped locally using only Python, Streamlit, and optionally a local LLM (Ollama + Gemma 3).

🚀 Features

Lead Management

Add leads via a Streamlit form

Import leads from CSV (headers: Name, Email, Company, UseCase, Budget, Phone)

Leads stored in memory (optionally persisted to JSON)

Qualification

Leads qualified if Budget ≥ Threshold (default: $5000, configurable in sidebar)

Qualified leads automatically generate a call request (JSON file in outbox/call_requests/)

Unqualified leads are marked and logged

Webhook Simulator (instead of Vapi)

Real systems would use a service like Vapi.ai
 for outbound calls + webhooks

Here we provide a Webhook Simulator tab to manually simulate call outcomes:

call.no_answer / call.unanswered → saves a follow-up email in outbox/emails/, marks lead as No Answer

call.completed / call.transcript_finalized / call.summary:

Saves transcript

If transcript includes “send a proposal” intent → generates a proposal, saves in proposals/, and attaches to a local “email” in outbox/emails/

Otherwise → logs “call completed, no proposal requested”

Proposal Generation

By default: static Markdown proposal template

If Ollama
 is installed and gemma3 pulled:

ollama pull gemma3


The system uses LangChain’s ChatOllama wrapper to generate tailored proposals based on lead details

Outbox Simulation

“Emails” saved as Markdown under outbox/emails/

“Call requests” saved as JSON under outbox/call_requests/

“Proposals” saved under proposals/

Notifications appended to outbox/notifications.log

UI Tabs

Leads: add, view, qualify, generate proposals manually

Import CSV: batch import leads

Calls: trigger simulated calls

Webhook Simulator: simulate outcomes instead of real Vapi webhook

Outbox Viewer: browse saved emails, proposals, and call requests

📂 Folder Structure
├── lead_automation_streamlit.py   # main app
├── data/
│   └── leads.json                 # persisted leads (optional)
├── outbox/
│   ├── emails/                    # saved emails as markdown
│   ├── call_requests/             # simulated outbound call requests (json)
│   └── notifications.log          # notification stream
└── proposals/                     # generated proposals (markdown)

⚙️ Installation & Run
Requirements

Python 3.9+

Streamlit

(Optional) Ollama
 with Gemma 3 model for real LLM proposals

Install dependencies
pip install streamlit pandas langchain langchain-community

Run app
streamlit run lead_automation_streamlit.py

Start Ollama (optional, for LLM proposals)
ollama serve
ollama pull gemma3

🧑‍💼 User Flow

Open the app → browser UI launches

Add leads via form or upload CSV

Qualify leads → if budget ≥ threshold, a call request file is created

Webhook Simulator → simulate call outcomes:

No Answer → follow-up email saved

Call Completed + Proposal Request → generate proposal, attach in local email

Review outputs in Outbox Viewer (emails, proposals, call requests)



🔧 Enhancements (Future Work)

Persistence:
Currently leads are stored in memory. Could extend to auto-save in data/leads.json or SQLite.

UI Improvements:
Add charts for qualified vs. unqualified leads, budgets, etc.

Real Integrations:

Replace Webhook Simulator with real Vapi webhooks

Replace Markdown emails with actual SendGrid or SMTP

Replace local JSON storage with Airtable, Postgres, or Firebase

Multi-LLM Support:
Allow switching between Ollama local models (Gemma, Llama 3, Mistral) for proposals.

Team Collaboration:
Deploy app on Streamlit Cloud or Docker with shared persistent storage for teams.
