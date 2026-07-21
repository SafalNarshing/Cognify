<div align="center">

# 🧠 Cognify

**An AI-assisted mental health platform combining clinical-style screening, cognitive science tasks, and NLP-driven journal analysis.**

[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-BERT--CNN-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Supabase](https://img.shields.io/badge/Supabase-auth%20%26%20db-3FCF8E?logo=supabase&logoColor=white)](https://supabase.com)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)](#-whats-not-yet-built)
[![Paper](https://img.shields.io/badge/Paper-KUSET%20Vol.18%20No.2-8A2BE2)](Cognify_KUSET.pdf)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<br/>

<img src="https://github.com/SafalNarshing/Cognify/blob/eaf97b86283082c9ee2988e6ba5659972f22f2e5/cognify/public/landing.png" alt="Cognify logo" width="420"/>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#-features)
- [Cognitive Tasks](#-cognitive-tasks)
- [NLP Model](#-nlp-model)
- [What's Not (Yet) Built](#-whats-not-yet-built)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Ethical Considerations](#-ethical-considerations)
- [Citation](#-citation)
- [License](#-license)

---

## Overview

Cognify pairs a short psychological screening with three well-established cognitive-psychology tasks (Flanker, Stroop, N-back) and a journaling feature analyzed by a custom-trained BERT-CNN model. It's the applied companion to the paper *["Cognify: Enhancement of Mental Conditions Using Cognitive Tools"](Cognify_KUSET.pdf)* (Pathak, KC, Kayastha, Pandey & Pathak — Kathmandu University Journal of Science, Engineering and Technology, Vol. 18, No. 2, Dec. 2024).

> **Disclaimer:** Cognify is a student research prototype, not a certified medical device. Its screening flow is inspired by — but is **not** — the licensed PHQ-9 / GAD-7 / ASRS instruments, and it has not undergone clinical trials. It is not a substitute for diagnosis or treatment by a licensed professional. If you or someone you know is in crisis, please contact a local emergency service or mental health helpline.

---

## Problem Statement

Anxiety, depression, and ADHD affect roughly 1 in 8 people worldwide, a burden that grew sharply after the COVID-19 pandemic. Access to personalized, scientifically grounded mental health support remains limited by stigma, cost, and a shortage of professionals — and existing digital tools (e.g., Woebot, Wysa) mostly offer generic CBT content or conversational check-ins without adapting to a user's ongoing psychological state or linking directly to their written self-expression.

Cognify explores whether a short screening step, standardized cognitive tasks used in psychology research, and NLP analysis of free-form journal entries can be brought together into one connected tool.

---

## ✨ Features

This repo has two parts: a Next.js web app (`cognify/`) and a Python NLP/chat service (`backend/`).

| Feature | Description |
|---|---|
| 🔐 **Auth & onboarding** | Email/password and Google sign-in (Supabase), followed by a profile intake form (age, gender, prior mental health history, daily routine) |
| 🧪 **Screening questionnaire** | A short custom questionnaire modeled on PHQ-9 / GAD-7 / ASRS-style screening, normalized into a profile category (e.g. depressive tendencies, neurodivergent patterns, cognitive decline risk, general wellness) |
| 🎮 **Cognitive tasks** | Three fully playable, timed tasks: Flanker, Emotional Stroop, and N-back — each scoring accuracy, reaction time, and errors |
| 🎯 **Task recommendation** | A one-time, rule-based mapping from the user's screening profile to the most relevant cognitive task |
| 📓 **Journaling** | Free-text journal with entry history; each entry is sent to the backend for NLP analysis and the results are stored with it |
| 💬 **AI chat & summaries** | A chat assistant and a "summarize my recent activity" feature, both backed by a locally hosted LLM |
| 📊 **Dashboard** | Visualizes cognitive task performance and screening status over time (Chart.js) |
| 🧘 **Mindfulness library** | Guided-audio UI; ships with a small starter set of intro tracks today |

---

## 🎮 Cognitive Tasks

| Task | Measures | Reference |
|---|---|---|
| **Flanker** | Interference suppression / attentional control | Eriksen & Eriksen, 1974 |
| **Emotional Stroop** | Response inhibition, emotional interference | Stroop, 1935 |
| **N-back** | Working memory | Kirchner, 1958 |

---

## 🧬 NLP Model

The backend runs a real, trained **hybrid BERT-CNN multi-task model** (`bert-base-uncased` + parallel CNN branches), not a stubbed response:

```
Input Text
    │
    ▼
[BERT Encoder] → contextual embeddings
    │
    ▼
[Conv1D branches, k = 2,3,4] → local feature extraction
    │
    ▼
[Pool + Dropout]
    │
    ├──► Sentiment Head            (positive / neutral / negative)
    ├──► Emotion Head              (27 GoEmotions labels, multi-label)
    ├──► Cognitive Distortion Head (10 distortion types)
    └──► Mental Health Head        (depression / anxiety / ADHD)
```

Trained weights (`bert_cnn_best_mh.pt`) are loaded at startup and served in real time via `/analyze` and `/analyzemulti`. A separate locally hosted `gemma3:1b` model (via Ollama + LangChain) powers the chat assistant and journal summarizer.

**Macro-F1 on public validation sets** (Reddit Mental Health Dataset, GoEmotions, Cognitive Distortion Detection Dataset):

| Task | Macro F1 |
|---|---|
| Mental health classification | **0.9907** |
| Sentiment analysis | 0.7357 |
| Cognitive distortion detection | 0.4783 |
| Multi-label emotion detection | 0.3018 |

These are dataset-level numbers for the NLP model alone, not a clinical outcome evaluation of Cognify as a whole.

---


## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4, Framer Motion, Chart.js |
| Auth & Data | Supabase (Postgres, Auth, SSR client) |
| Backend | Python, FastAPI, Uvicorn |
| NLP Model | PyTorch, HuggingFace Transformers (BERT) + custom CNN heads |
| Conversational AI | Ollama (`gemma3:1b`) via LangChain |

---

## 📁 Project Structure

```
Cognify/
├── cognify/                      # Next.js frontend
│   ├── app/
│   │   ├── api/                  # auth, profile, questionnaire, journal, chat, task-result routes
│   │   ├── auth/                 # Login / signup pages
│   │   ├── onboarding/
│   │   │   ├── info/             # Profile intake
│   │   │   ├── screening/        # Screening questionnaire
│   │   │   ├── profiling/        # Condition classification result
│   │   │   ├── assessment/       # Recommended cognitive task
│   │   │   └── dashboard/        # Dashboard, journal, chat, mindfulness
│   │   │       ├── flanker/
│   │   │       ├── stroop/
│   │   │       └── nback/
│   │   └── components/           # Screening data, chat widget, mindfulness UI, progress chart
│   ├── lib/                       # Supabase client, middleware helpers
│   └── middleware.ts
│
└── backend/                       # FastAPI NLP + chat service
    ├── app.py                     # /analyze, /analyzemulti, /chat, /generate, /summarizer
    ├── BERT_CNN.py                 # Single-task BERT-CNN classifier
    ├── BERT_CNN_multi.py           # Multi-task BERT-CNN classifier
    ├── inference_module.py         # Single-task inference
    ├── preprocess_multi.py         # Multi-task inference
    ├── bert_cnn_best_mh.pt          # Trained model weights
    └── requirements.txt
```

---

## 🚀 Getting Started

### Frontend

```bash
cd cognify
npm install
cp .env.example .env.local
# fill in NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY, JOURNAL_API_BASE
npm run dev
```
Runs at `http://localhost:3000`.

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Requires [Ollama](https://ollama.com/) running locally with the chat model pulled:

```bash
ollama pull gemma3:1b
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Runs at `http://localhost:8000` (interactive docs at `/docs`). Point the frontend's `JOURNAL_API_BASE` (and chat/summarizer config) at this address.

---

## ⚖️ Ethical Considerations

- Cognify is a **supplementary tool**, not a replacement for professional mental health care.
- Journal entries and screening answers are sensitive data — treat any deployment's Supabase project and environment secrets accordingly.
- The screening flow uses simplified, non-clinical question sets inspired by (but not identical to) PHQ-9, GAD-7, and ASRS — do not present scores from this app as clinical diagnoses.



---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

Note this covers the Cognify code only. The pre-trained model weights (`bert_cnn_best_mh.pt`) were trained on third-party datasets (Reddit Mental Health Dataset, GoEmotions, Cognitive Distortion Detection Dataset) that carry their own licensing/usage terms — review those before any commercial or clinical use.

---

<div align="center">

If this project interests you, consider giving it a ⭐

</div>
