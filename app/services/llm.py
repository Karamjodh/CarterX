import asyncio
from app.core.config import settings
async def generate_report(prompt : str, model: str = None) -> dict:
    """
    Sends a prompt to the chosen LLM and returns the response.

    Args:
        prompt:  The full prompt text to send
        model:   Which LLM to use — "gemini", "openai", or "anthropic"
                 Defaults to whatever DEFAULT_LLM is set to in config

    Returns:
        {
            "text":         the response text,
            "model_used":   which model actually ran,
            "input_tokens": how many tokens were sent,
            "output_tokens": how many tokens came back
        }
    """
    chosen_model = model or settings.DEFAULT_LLM
    if chosen_model == "groq":
        return await _call_groq(prompt)
    elif chosen_model == "gemini":
        return await _call_gemini(prompt)
    elif chosen_model == "openai":
        return await _call_openai(prompt)
    elif chosen_model == "anthropic":
        return await _call_anthropic(prompt)
    else:
        raise ValueError(f"Unknown model '{chosen_model}'."
                         f"Choose from: 'groq','gemini', 'openai','anthropic'")
    
async def _call_groq(prompt: str) -> dict:
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in your .env file")

    def _sync_call():
        from groq import Groq
        client   = Groq(api_key=settings.GROQ_API_KEY)
        response = client.chat.completions.create(
            model      = "llama-3.3-70b-versatile",
            max_tokens = 2048,
            messages   = [
                {"role": "system", "content": "You are a senior marketing strategist. Give clear, specific, data-driven recommendations to the user. Be concise and actionable in your advice."},
                {"role": "user",   "content": prompt}
            ]
        )
        return {
            "text":         response.choices[0].message.content,
            "model_used":   "llama-3.3-70b-versatile (Groq)",
            "input_tokens":  response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    return await asyncio.to_thread(_sync_call)
async def _call_gemini(prompt: str) -> dict:
    if not settings.GEMINI_API_KEY:
        raise ValueError("Gemini_API_KEY is not set in your .env file")

    import google.generativeai as genai # Software development kit SDK
    genai.configure(api_key = settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return {
        "text" : response.text,
        "model_used" : "gemini-2.0-flash",
        "input_tokens" : None,
        "output_tokens" : None,
    }    

async def _call_openai(prompt: str) -> dict:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in your .env file")
    
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key = settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model = "gpt-4o-mini",
        max_tokens = 2048,
        messages = [
            {"role" : "user",
             "content" : prompt}
        ]
    )
    return {
        "text" : response.choices[0].message.content,
        "model_used" : "gpt-4o-mini",
        "input_tokens" : response.usage.prompt_tokens,
        "output_tokens" : response.usage.completion_tokens,
    }

async def _call_anthropic(prompt : str) -> dict:
    if not settings.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set in your .env file")
    import anthropic
    client = anthropic.Anthropic(api_key = settings.ANTHROPIC_API_KEY)
    response = client.message.create(
        model = "claude-sonnet-4-20250514",
        max_tokens = 2048,
        messages = [
            {"role" : "user",
             "content" : prompt}
        ]
    )
    return {
        "text" : response.content[0].text,
        "model_used" : "claude-sonnet-4-20250514",
        "input_token" : response.usage.input_tokens,
        "output_tokens" : response.usage.output_tokens,
    }
