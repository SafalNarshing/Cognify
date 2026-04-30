from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference_module import get_prediction
from preprocess_multi import MultiTaskInference
from langchain_ollama import ChatOllama
from typing import List, Dict, Optional
from fastapi.responses import JSONResponse
from typing import Any
import re

llm = ChatOllama(model="gemma3:1b", temperature=0.1)
# llm = ChatOllama(model="cniongolo/biomistral:latest", temperature=0.2)

app = FastAPI(title="Mental Health Analysis API")

multi_inferencer = MultiTaskInference(model_path="C:\\Coding\\cognify\\models\\bert_cnn_multitask2.pth")

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "http://localhost:3000/onboarding/dashboard",
    "http://127.0.0.1:3000",
    "http://10.10.254.183:3000",
    "https://your-prod-domain",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # includes OPTIONS
    allow_headers=["*"],
)

class JournalRequest(BaseModel):
    text: str

class Message(BaseModel):
    role: str  # "user" or "model"
    content: str

class ChatRequest(BaseModel):
    text: str
    history: Optional[List[Message]] = []

class JournalItem(BaseModel):
    id: Optional[str] = None
    user_id: Optional[str] = None
    content: str
    analysis: Optional[str] = None
    sentiment: Optional[str] = None
    detected_emotions: Optional[str] = None
    cognitive_distortion: Optional[str] = None
    dominant_prediction: Optional[str] = None

class UserProfile(BaseModel):
    fullname: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    prior_mental_health_issue: Optional[str] = None
    daily_routine: Optional[str] = None

# class SummarizerRequest(BaseModel):
#     journal: List[JournalItem]
#     user_profile: UserProfile
#     user_status: Dict[str, Any] = {}

class StroopData(BaseModel):
    rt_neutral_avg: float
    rt_emotional_avg: float
    accuracy_neutral: float
    accuracy_emotional: float
    interference_score: float

class FlankerData(BaseModel):
    score: float
    accuracy: float
    avg_reaction_time_ms: float
    misses: int
    false_positives: int

class NBackData(BaseModel):
    accuracy: float
    avgReactionTime: float
    hits: int
    misses: int
    totalTrials: int

class SummarizerRequest(BaseModel):
    journal: List[JournalItem]
    user_profile: UserProfile
    user_status: Dict[str, Any] = {}
    # Cognitive Data Inputs
    stroop: Optional[List[StroopData]] = None
    flanker: Optional[List[FlankerData]] = None
    nback: Optional[List[NBackData]] = None

# --- PROMPT BUILDERS ---

def build_cognitive_context(stroop, flanker, nback):
    """Formats the raw cognitive data into a readable summary for the LLM."""
    context = ""
    if stroop and len(stroop) > 0:
        s = stroop[-1]
        context += f"\n- Stroop Task: Interference Score of {s.interference_score}ms (Neutral RT: {s.rt_neutral_avg}ms, Emotional RT: {s.rt_emotional_avg}ms). Accuracy: {s.accuracy_emotional}%."
    if flanker and len(flanker) > 0:
        f = flanker[-1]
        context += f"\n- Flanker Task: Attention Accuracy of {f.accuracy}% with an average reaction time of {f.avg_reaction_time_ms}ms."
    if nback and len(nback) > 0:
        n = nback[-1]
        context += f"\n- N-Back Task: Working Memory Accuracy of {n.accuracy}% across {n.totalTrials} trials."
    
    return context if context else "No recent cognitive task data available."

def build_summarizer_system_prompt(profile: UserProfile | Dict[str, Any], status: Dict[str, Any], stroop, flanker, nback) -> str:
    prior_history = profile.prior_mental_health_issue or "no known history"
    baseline_status = status.get("baseline_state", "unknown")
    severity_level = status.get("severity_level", "not assessed")
    sentiment = status.get("journal_sentiment", "neutral")
    detected_emotions = status.get("detected_emotions", "none detected")
    cognitive_distortion = status.get("cognitive_distortion", "none noted")
    user_name = profile.fullname or "the user"
    age_text = f"{profile.age}" if profile.age is not None else "their age group"
    gender_text = profile.gender if profile.gender else "their identity"
    routine = profile.daily_routine or "no routine provided"

    # cognitive_performance = build_cognitive_context(stroop, flanker, nback)

    def flatten(d: Dict[str, Any], prefix: str = "") -> List[str]:
        parts = []
        for k, v in d.items():
            if isinstance(v, dict):
                parts.extend(flatten(v, prefix=f"{prefix}{k}."))
            else:
                parts.append(f"{prefix}{k}: {v}")
        return parts

    status_summary = " | ".join(flatten(status)) if status else ""
    return f"""
You are a licensed, highly experienced mental health therapist and motivational coach. 
Your role is to support {user_name} with empathy, encouragement, and practical life guidance, without diagnosing or medicalizing unless asked.

Speak in an age-appropriate, respectful tone for a {age_text}-year-old who identifies as {gender_text}. 
Consider their daily routine ({routine}), prior mental health history ({prior_history}), recent emotional state ({sentiment}, {detected_emotions}), and thinking patterns ({cognitive_distortion}).

Interpret cognitive task performance qualitatively as signals of focus or emotional load, never as pathology or scores.
Lead with validation, highlight strengths, normalize fluctuation, and offer gentle, optional suggestions tied to daily life.

Keep responses warm, supportive, non-clinical, and hopeful.
Do not mention tests, scores, models, or internal logic.

**Response guidelines:**
- Each response must be limited to atmost 50 words.
- Maintain a compassionate, human, and reassuring tone.
- Focus on fostering understanding, empowerment, and emotional comfort.
"""

#     return f"""
# ROLE & IDENTITY
# You are a highly experienced, licensed mental health therapist and motivational coach.
# Your role is to support emotional well-being through empathy, reflection, and practical life guidance grounded in cognitive science.
# You are NOT here to diagnose, label, or medicalize the user unless they explicitly request it.

# You interpret cognitive task performance as signals of mental state, not as pathology.

# USER CONTEXT (CONFIDENTIAL)

# Name: {user_name}

# Age: {age_text}

# Gender identity: {gender_text}

# Prior mental health history: {prior_history}

# Daily routine: {routine}

# CURRENT WELL-BEING SNAPSHOT

# Overall baseline state: {baseline_status}

# Current risk/severity level: {severity_level}

# Recent journal sentiment: {sentiment}

# Detected emotions: {detected_emotions}

# Noted thinking pattern: {cognitive_distortion}


# This data may include reaction time, accuracy, interference scores, or error counts from tasks such as Stroop, Flanker, or N-Back.
# You must interpret this data qualitatively, never report raw numbers.

# INTERPRETATION FRAMEWORK (ONE-SHOT GUIDANCE)

# Example A — Stroop Task
# Input (internal):
# User shows low interference and stable reaction time across neutral and emotional words.

# Correct interpretation style:
# “The user demonstrates strong emotional regulation and attentional control. Emotional content did not ‘capture’ attention, suggesting calm focus even under potentially distracting stimuli.”

# Language to user:
# “You seem able to stay steady and focused, even when emotionally charged thoughts come up. That’s a real strength.”

# Example B — Flanker Task
# Input (internal):
# User shows low accuracy and frequent false positives.

# Correct interpretation style:
# “This pattern suggests difficulty filtering distractions and resolving interference. The user’s executive window may feel narrow today, leading to impulsivity or scattered attention.”

# Language to user:
# “Your mind may feel a bit crowded today, like everything is competing for attention. That can be tiring, and it’s okay to slow things down.”

# Use these examples as a template for interpreting all cognitive task data.

# PROGRESS AWARENESS
# If prior cognitive or emotional data exists:

# Briefly acknowledge progress, stability, or fluctuation.

# Frame change as normal and non-judgmental.

# Reinforce effort, consistency, or self-awareness.

# Example tone:
# “Compared to earlier check-ins, you’re showing more steadiness in focus.”
# or
# “Today looks a little heavier than usual — that happens, especially when life demands more.”

# Never compare the user to norms or other people.

# THERAPEUTIC APPROACH

# Speak directly to {user_name} using warm, respectful, age-appropriate language.

# Lead with empathy before insight.

# Normalize emotional and cognitive variability.

# Tie reflections gently to their daily routine: {routine}.

# Emphasize strengths, adaptability, and resilience.

# Avoid diagnostic labels, scores, or technical task names.

# GUIDANCE STYLE

# Keep tone calm, human, and supportive.

# Suggestions must feel optional, not prescriptive.

# If focus or interference is low, suggest grounding, pacing, or shorter attention windows.

# Avoid urgency unless risk level is high.

# RESPONSE FORMAT

# 2–3 short paragraphs maximum.

# Optionally include a gentle “small steps for today” list (2–3 items).

# Do NOT use markdown, emojis, role labels, or formatting symbols.

# Do NOT mention raw scores, test names, or internal model logic.

# RESPONSE MUST BE BETWEEN 50 AND 100 WORDS.
# """

def build_summarizer_user_message(journal: List[JournalItem], profile: UserProfile) -> str:
    name = profile.fullname or "the user"
    lines = [
        f"Profile: name={name}, age={profile.age or 'unknown'}, routine={profile.daily_routine or 'unspecified'}.",
        "Recent journal notes:"
    ]
    max_items = min(len(journal), 8)
    for i, item in enumerate(journal[:max_items], start=1):
        content = item.content.strip()
        analysis = (item.analysis or "").strip()
        line = f"{i}. {content}"
        if analysis:
            line += f" | analysis: {analysis}"
        lines.append(line)
    lines.append("Please respond supportively and focus on improvement and small steps.")
    return "\n".join(lines)

def extract_text(resp) -> str:
    """
    Return only the textual content from a LangChain/Ollama response.
    Falls back to string if unexpected type, and removes common prefixes.
    """
    try:
        text = getattr(resp, "content", None)
        if text is None and isinstance(resp, dict):
            text = resp.get("content") or resp.get("text")
        # Fallback for any other shapes
        if text is None:
            text = str(resp)

        # Cleanup: trim and drop leading "AI:"/"Assistant:" if present
        text = text.strip()
        text = re.sub(r'^(AI|Assistant)\s*:\s*', '', text, flags=re.I)
        return text
    except Exception:
        return str(resp).strip()
    
def clean_llm_response(text: str) -> str:
    if not text:
        return ""

    # 1. Remove markdown emphasis symbols (*, **, _, __) but keep content
    text = re.sub(r'(\*{1,2}|_{1,2})(.*?)\1', r'\2', text)

    # 2. Remove remaining standalone markdown/special characters
    #    Includes *, _, /, \, `, ~
    text = re.sub(r'[*_/\\`~]', '', text)

    # 3. Normalize newlines, tabs, multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


@app.post("/analyze")
async def analyze_journal(request: JournalRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Journal text cannot be empty")
    
    try:
        prediction = get_prediction(request.text)
        return {"Mental Health": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/analyzemulti")
async def analyze_multi(request: JournalRequest):
    """New Multi-Task endpoint: Mental Health, Sentiment, Emotion, Distortion."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        results = multi_inferencer.predict(request.text)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/generate")
async def generate_response(request: JournalRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    try:
        prompt = f"Provide supportive and empathetic feedback for the following journal entry:\n\n{request.text}\n\nResponse:"
        response = llm.predict(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

def format_gemma_3_prompt(system_instruction: str, history: List[Message], current_user_msg: str) -> str:
    """Formats the prompt using Gemma 3's turn-based control tokens."""
    prompt = f"<start_of_turn>developer\n{system_instruction}<end_of_turn>\n"
    
    # Interleave previous conversation history
    for msg in history:
        role = "user" if msg.role == "user" else "model"
        prompt += f"<start_of_turn>{role}\n{msg.content}<end_of_turn>\n"
    
    # Add current user input
    prompt += f"<start_of_turn>user\n{current_user_msg}<end_of_turn>\n"
    prompt += "<start_of_turn>model\n"
    return prompt

@app.post("/chat")
async def chat_with_ai(data: ChatRequest):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    try:
        system_prompt = """
        # ROLE
        You are a compassionate Mental Health Assistant.

        # GUIDELINES
        - Be brief, empathetic, and validating.
        - Prioritize active listening.
        - If a user is in crisis, prioritize safety and suggest professional help.
        - Avoid clinical jargon; speak like a supportive friend.

        # CONSTRAINTS
        - Do not give specific medical prescriptions.
        - Keep responses under 3 paragraphs unless asked for detail.
        """

        # 3. Construct the prompt with Memory
        full_prompt = format_gemma_3_prompt(system_prompt, data.history, data.text)
        
        response = llm.invoke(full_prompt) 
        
        content = response.content

        filtered = clean_llm_response(content)

        return filtered
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/summarizer")
async def summarizer(req: SummarizerRequest):
    # Validate minimal input
    if not req.journal or all(not (j.content or "").strip() for j in req.journal):
        raise HTTPException(status_code=400, detail="At least one journal entry with content is required.")

    try:
        system_prompt = build_summarizer_system_prompt(req.user_profile, req.user_status, req.stroop,
            req.flanker,
            req.nback)
        user_msg = build_summarizer_user_message(req.journal, req.user_profile)

        full_prompt = format_gemma_3_prompt(system_prompt, [], user_msg)
        response = llm.invoke(full_prompt)

        content = response.content
        filtered = clean_llm_response(content)
        return filtered
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarizer error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)

