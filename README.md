# 🧠 Cognify — Enhancement of Mental Conditions Using Cognitive Tools

> A personalized, AI-driven mental health platform combining clinical screening, cognitive science-based interventions, and NLP-driven journal analysis.

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Tech Stack](#tech-stack)
5. [Prerequisites](#prerequisites)
6. [Project Structure](#project-structure)
7. [Setup & Installation](#setup--installation)
   - [Frontend (Next.js)](#frontend-nextjs)
   - [Backend (Python / FastAPI)](#backend-python--fastapi)
8. [Environment Variables](#environment-variables)
9. [Usage](#usage)
10. [NLP Module & ML Pipeline](#nlp-module--ml-pipeline)
11. [Cognitive Tasks](#cognitive-tasks)
12. [Adaptive Task Selection Algorithm](#adaptive-task-selection-algorithm)
13. [Ethical Considerations](#ethical-considerations)
14. [Research & Citations](#research--citations)
15. [Contributing](#contributing)
16. [License](#license)

---

## Overview

Mental health disorders like **Anxiety**, **Depression**, and **ADHD** affect nearly 1 in 8 people globally (970 million+ as of 2019). Cognify addresses the gap in personalized, clinically grounded, and accessible digital mental health tools.

Cognify is an integrative platform that:

- Screens users using validated psychiatric instruments (PHQ-9, GAD-7, ASRS)
- Recommends condition-specific cognitive tasks (Flanker, Stroop, N-back)
- Analyzes user journal entries with a hybrid **BERT-CNN NLP model**
- Continuously adapts recommendations through a closed-loop feedback mechanism

---

## Features

| Feature | Description |
|---|---|
| 🧪 **Clinical Screening** | PHQ-9, GAD-7, ASRS assessments at onboarding |
| 🎮 **Cognitive Tasks** | Flanker Task, Stroop Task, N-back Task mapped to conditions |
| 📓 **Journaling** | Free-text journal with NLP-driven sentiment & distortion analysis |
| 📊 **Dashboard** | Personalized progress tracking across tasks and journal trends |
| 🔁 **Adaptive Feedback Loop** | Recommendations evolve based on performance + journal insights |
| 🧘 **Mindfulness Tools** | Guided breathing and attention redirection exercises |
| 🔒 **Privacy-First** | GDPR/HIPAA-aligned, PII anonymization, encrypted data |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User (Browser/App)                 │
└───────────────────────────┬─────────────────────────────┘
                            │
            ┌───────────────▼───────────────┐
            │       Frontend (Next.js)       │
            │  - Onboarding & Assessments   │
            │  - Cognitive Game UI          │
            │  - Journaling Interface       │
            │  - Dashboard & Progress       │
            └───────────────┬───────────────┘
                            │ REST API
            ┌───────────────▼───────────────┐
            │      Backend (Python/FastAPI)  │
            │  - Assessment Scoring         │
            │  - User Profile Management    │
            │  - NLP Pipeline (BERT-CNN)    │
            │  - Adaptive Task Selector     │
            │  - Mindfulness Recommender    │
            └───────────────┬───────────────┘
                            │
            ┌───────────────▼───────────────┐
            │          ML / NLP Layer        │
            │  - Sentiment Analysis         │
            │  - Emotion Classification     │
            │  - Cognitive Distortion Det.  │
            │  - Mental Health Classifier   │
            └───────────────────────────────┘
```

**System Workflow:**

```
User Onboarding
     │
     ▼
Mental Health Screening (PHQ-9 / GAD-7 / ASRS)
     │
     ▼
Mental Condition Profiling (Anxiety / Depression / ADHD)
     │
     ├──────────────────────────────────┐
     ▼                                  ▼
Cognitive Tasks                     Journal Entry
(Flanker / Stroop / N-back)         (Free text)
     │                                  │
     ▼                                  ▼
Task Result Analysis            Sentiment Analysis
     │                          (BERT-CNN Model)
     └──────────────┬───────────────────┘
                    ▼
         User's Personalized Dashboard
         + Adaptive Recommendations
         + Mindfulness & Self-care Tools
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js (TypeScript), React, CSS Modules |
| **Backend** | Python, FastAPI |
| **NLP Model** | BERT + CNN hybrid architecture (PyTorch / HuggingFace Transformers) |
| **ML Training** | Scikit-learn, PyTorch |
| **Data** | Reddit Mental Health Dataset, GoEmotions, Cognitive Distortion Dataset |

---

## Prerequisites

Make sure the following are installed:

**For Frontend:**
- [Node.js](https://nodejs.org/) v18+ 
- npm or yarn

```bash
node -v
npm -v
```

**For Backend:**
- Python 3.9+
- pip

```bash
python --version
pip --version
```

---

## Project Structure 

<!-- ```
Cognify/
├── cognify/                    # Next.js frontend application
│   ├── app/                    # App router pages
│   ├── components/             # Reusable React components
│   │   ├── assessments/        # PHQ-9, GAD-7, ASRS forms
│   │   ├── games/              # Flanker, Stroop, N-back task UIs
│   │   ├── journal/            # Journaling interface
│   │   └── dashboard/          # Progress tracking UI
│   ├── public/                 # Static assets
│   ├── styles/                 # Global CSS
│   ├── package.json
│   └── tsconfig.json
│
├── Backend/                    # Python FastAPI backend
│   ├── main.py                 # FastAPI application entry point
│   ├── models/                 # Pydantic data models
│   ├── routers/                # API route handlers
│   │   ├── assessment.py       # Screening endpoints
│   │   ├── journal.py          # Journal analysis endpoints
│   │   └── tasks.py            # Cognitive task endpoints
│   ├── nlp/                    # NLP pipeline
│   │   ├── bert_cnn.py         # BERT-CNN model definition
│   │   ├── sentiment.py        # Sentiment analysis
│   │   ├── distortion.py       # Cognitive distortion detection
│   │   └── classifier.py       # Mental health classification
│   ├── adaptive/               # Adaptive recommendation engine
│   │   └── task_selector.py    # Algorithm 1 implementation
│   ├── requirements.txt
│   └── .env.example
│
├── README.md
└── .gitattributes
```  -->

<!-- --- -->

## Setup & Installation

### Frontend (Next.js)

1. **Clone the repository**

```bash
git clone https://github.com/SafalNarshing/Cognify.git
cd Cognify
```

2. **Navigate to the frontend directory**

```bash
cd cognify
```

3. **Install dependencies**

```bash
npm install
```

4. **Set up environment variables**

```bash
cp .env.example .env.local
# Edit .env.local with your backend API URL
```

5. **Run the development server**

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

6. **Build for production**

```bash
npm run build
npm start
```

---

### Backend (Python / FastAPI)

1. **Navigate to the backend directory**

```bash
cd Backend
```

2. **Create and activate a virtual environment**

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Download pre-trained models** *(if not bundled)*

```bash
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased')"
```

6. **Run the backend server**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`  
Interactive API docs: `http://localhost:8000/docs`

---

## Environment Variables

### Frontend (`cognify/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=Cognify
```

### Backend (`backend/.env`)

```env
# Application
APP_NAME=Cognify
DEBUG=True
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=sqlite:///./cognify.db
# For PostgreSQL: DATABASE_URL=postgresql://user:password@localhost/cognify

# ML Models
MODEL_PATH=./models/bert_cnn_checkpoint.pt
HF_MODEL_NAME=bert-base-uncased

# CORS
ALLOWED_ORIGINS=http://localhost:3000
```

---

## Usage

### 1. Onboarding & Assessment

When a new user registers, they complete three validated assessments:

| Scale | Condition | Score Range |
|---|---|---|
| **PHQ-9** | Depression | 0–27 |
| **GAD-7** | Anxiety | 0–21 |
| **ASRS** | ADHD | 0–24 |

The system classifies the user into one of: `anxiety`, `depression`, or `adhd`.

### 2. Cognitive Tasks

Based on classification, users are directed to condition-specific tasks:

| Condition | Primary Task | Secondary Task | Target |
|---|---|---|---|
| Anxiety | Flanker Task | Breathing exercises | Attention control |
| Depression | N-back Task | Behavioral activation | Cognitive stimulation |
| ADHD | N-back Task | Attention tasks | Working memory |

### 3. Journaling

Users can write freely about their thoughts and feelings. Each entry is analyzed for:
- **Sentiment polarity** (positive / negative / neutral)
- **Emotional state** (multi-label, GoEmotions taxonomy)
- **Cognitive distortions** (10 types including catastrophizing, black-and-white thinking)
- **Mental health indicators** (anxiety/depression/ADHD linguistic markers)

### 4. Dashboard

The dashboard visualizes:
- Task performance over time (accuracy, reaction time)
- Sentiment trend from journal entries
- Adaptive difficulty progression
- Recommended mindfulness exercises

---

## NLP Module & ML Pipeline

Cognify uses a **hybrid BERT-CNN architecture** trained on three public datasets:

| Dataset | Purpose | Source |
|---|---|---|
| Reddit Mental Health Dataset | Mental health classification | Zenodo |
| GoEmotions | Sentiment & emotion classification | Google Research |
| Cognitive Distortion Dataset | Distortion type classification | Kaggle |

### Model Performance (Macro F1-Score)

| Task | F1-Score |
|---|---|
| Mental Health Classification | **0.9907** |
| Sentiment Analysis | 0.7357 |
| Cognitive Distortion Detection | 0.4783 |
| Multi-label Emotion Classification | 0.3018 |

### Architecture Overview

```
Input Text
    │
    ▼
[BERT Encoder] → Contextual embeddings (768-dim)
    │
    ▼
[CNN Layers] → Local feature extraction
    │
    ▼
[Pooling + Dense]
    │
    ├──► Sentiment Head
    ├──► Emotion Head (multi-label)
    ├──► Cognitive Distortion Head
    └──► Mental Health Classification Head
```

---

## Cognitive Tasks

### Flanker Task
Tests **attentional control** and **interference suppression**. Users identify the direction of a central arrow surrounded by flanking arrows pointing the same (congruent) or opposite (incongruent) directions.

### Stroop Task
Tests **response inhibition**. Users identify the ink color of color words, where word meaning and ink color may conflict (e.g., the word "RED" printed in blue).

### N-back Task
Tests **working memory**. Users monitor a sequence of stimuli and respond when the current stimulus matches one presented `n` steps earlier.

Difficulty adapts based on the user's historical performance metrics.

---

## Adaptive Task Selection Algorithm

The core recommendation engine (Algorithm 1 from the paper):

```
Input:  Journal_Entry, User_Profile, Performance_History
Output: Calibrated_Tasks

Phase 1 — Feature Extraction (BERT-CNN)
  → Extract: sentiment, emotions, mh_indicators, cognitive_distortions

Phase 2 — Condition-Specific Mapping
  if condition == "anxiety":
      if anxiety_markers detected → Flanker + Breathing exercises
  if condition == "depression":
      if negative_sentiment detected → N-back + Behavioral activation
  if condition == "adhd":
      → N-back (working memory) + Sustained attention tasks

Phase 3 — Difficulty Calibration
  for each recommended task:
      adjust difficulty based on history.performance

Return calibrated_tasks
```

---

## Ethical Considerations

Cognify implements the following safeguards:

- **Informed Consent** — Explicit consent at registration; users can withdraw at any time
- **Data Privacy** — Industry-standard encryption at rest and in transit; GDPR & HIPAA aligned
- **Anonymization** — PII stored separately; automated PII removal in NLP pipelines
- **Bias Mitigation** — Regular model audits across demographic groups; Explainable AI (XAI)
- **Clinical Risk Management** — Cognify is a **supplementary tool, not a replacement** for professional care. Automated alerts are triggered for severe distress or suicidal ideation, prompting referral to helplines or emergency services

> ⚠️ **Disclaimer:** Cognify is not a medical device and does not replace professional psychiatric care. If you are experiencing a mental health crisis, please contact a licensed professional or emergency services.





