from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import asyncio
import os
import logging
import requests
import tempfile
import fitz  # Only for PDF text extraction
from dotenv import load_dotenv

# Minimal dependencies - no ML libraries!
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Cloud-Only Insurance Assistant", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class HackathonRequest(BaseModel):
    documents: str
    questions: List[str]


def extract_pdf_text_only(pdf_url: str) -> str:
    """Lightweight PDF extraction - no ML processing"""
    try:
        logger.info(f"[kg290] Downloading PDF: {pdf_url[:50]}...")

        response = requests.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name

        doc = fitz.open(temp_path)
        text = "\n\n".join([page.get_text() for page in doc])
        doc.close()
        os.unlink(temp_path)

        logger.info(f"[kg290] Extracted {len(text)} characters")
        return text

    except Exception as e:
        logger.error(f"[kg290] PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")


def find_relevant_sections(full_text: str, query: str, max_sections: int = 5) -> str:
    """Smart text-based section finding - no ML required"""

    # Extract keywords from query
    query_words = [word.lower().strip('.,!?') for word in query.split()
                   if len(word) > 3 and word.lower() not in ['what', 'when', 'where', 'how', 'the', 'and', 'for']]

    # Split document into logical sections
    sections = []
    current_section = ""

    for line in full_text.split('\n'):
        if line.strip():
            # New section if line looks like a header
            if (line.isupper() or
                    any(word in line.lower() for word in ['section', 'clause', 'article', 'chapter']) or
                    len(line) < 100):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'

    if current_section.strip():
        sections.append(current_section.strip())

    # Score sections by keyword relevance
    scored_sections = []
    for section in sections:
        score = sum(1 for word in query_words if word in section.lower())
        if score > 0:
            scored_sections.append((section, score))

    # Sort by relevance and take top sections
    scored_sections.sort(key=lambda x: x[1], reverse=True)
    relevant_sections = [section for section, _ in scored_sections[:max_sections]]

    return "\n\n---\n\n".join(relevant_sections)


async def cloud_only_analysis(query: str, pdf_text: str) -> Dict[str, Any]:
    """Pure cloud-based analysis using Groq API"""

    try:
        # Find relevant context without ML
        relevant_context = find_relevant_sections(pdf_text, query)

        # If context is too long, truncate intelligently
        max_context_chars = 100000  # Leave room for query and response
        if len(relevant_context) > max_context_chars:
            relevant_context = relevant_context[:max_context_chars] + "\n\n[Context truncated...]"

        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": """You are an expert insurance policy analyst. Provide accurate, detailed answers based solely on the policy document provided.

RESPONSE FORMAT:
{
    "answer": "detailed and specific answer",
    "confidence": "high|medium|low",
    "supporting_clauses": ["specific clause references"],
    "reasoning": "step-by-step explanation",
    "conditions": ["any relevant conditions or limitations"],
    "additional_notes": "important contextual information"
}

INSTRUCTIONS:
- Base answers only on the provided policy document
- Include specific clause numbers when available
- Explain your reasoning clearly
- Note any conditions or limitations
- If information is not in the document, state this clearly"""
                },
                {
                    "role": "user",
                    "content": f"""
QUESTION: {query}

POLICY DOCUMENT:
{relevant_context}

Analyze this question based on the policy document and provide your response in the specified JSON format."""
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 2000
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"[kg290] Sending query to Groq API...")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()["choices"][0]["message"]["content"]

        try:
            parsed_result = json.loads(result)
            return {
                "answer": parsed_result.get("answer", "Unable to determine"),
                "confidence": parsed_result.get("confidence", "medium"),
                "supporting_clauses": parsed_result.get("supporting_clauses", []),
                "reasoning": parsed_result.get("reasoning", ""),
                "conditions": parsed_result.get("conditions", []),
                "model_used": "llama3-70b-8192",
                "processing_type": "cloud_only",
                "context_length": len(relevant_context),
                "success": True
            }
        except json.JSONDecodeError:
            return {
                "answer": result,
                "confidence": "medium",
                "model_used": "llama3-70b-8192",
                "processing_type": "cloud_only_fallback",
                "success": True
            }

    except Exception as e:
        logger.error(f"[kg290] Cloud analysis failed: {str(e)}")
        return {
            "answer": f"Analysis failed: {str(e)}",
            "confidence": "low",
            "success": False
        }


@app.get("/")
def root():
    return {
        "message": "Cloud-Only Insurance Assistant v4.0",
        "status": "running",
        "architecture": "Pure cloud processing",
        "memory_usage": "< 100MB",
        "dependencies": ["FastAPI", "Requests", "PyMuPDF"],
        "user": "kg290",
        "timestamp": "2025-07-30 16:18:28"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "processing_type": "cloud_only",
        "api_available": bool(api_key),
        "timestamp": "2025-07-30 16:18:28"
    }


@app.post("/ask")
async def ask_query(req: QueryRequest):
    """Development endpoint with local PDF"""
    try:
        pdf_path = "data/Dataset1.pdf"

        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF file not found")

        with open(pdf_path, 'rb') as file:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "\n\n".join([page.get_text() for page in doc])
            doc.close()

        result = await cloud_only_analysis(req.query, text)

        return {
            "response": result.get("answer", "No answer generated"),
            "confidence": result.get("confidence", "medium"),
            "supporting_clauses": result.get("supporting_clauses", []),
            "reasoning": result.get("reasoning", ""),
            "processing_stats": {
                "type": "cloud_only",
                "model": result.get("model_used", "unknown"),
                "context_length": result.get("context_length", 0),
                "memory_usage": "minimal"
            },
            "user": "kg290"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hackrx/run")
async def hackathon_endpoint(request: HackathonRequest):
    """Optimized hackathon endpoint - pure cloud processing"""
    try:
        logger.info(f"[kg290] Cloud-only processing: {len(request.questions)} questions")

        # Extract PDF text
        pdf_text = extract_pdf_text_only(request.documents)

        # Process all questions
        answers = []

        for i, question in enumerate(request.questions):
            logger.info(f"[kg290] Processing question {i + 1}/{len(request.questions)}")

            # Small delay to prevent rate limiting
            if i > 0:
                await asyncio.sleep(0.5)

            result = await cloud_only_analysis(question, pdf_text)
            answer = result.get("answer", "Unable to determine from the policy document.")
            answers.append(answer)

        logger.info(f"[kg290] All questions processed successfully")

        return {"answers": answers}

    except Exception as e:
        logger.error(f"[kg290] Hackathon endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info(f"[kg290] Starting Cloud-Only Insurance Assistant")
    uvicorn.run(app, host="0.0.0.0", port=8000)