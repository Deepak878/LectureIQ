import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "gemini"


def get_llm_client():
    """initializing llm client based on LLM_PROVIDER"""
    if LLM_PROVIDER == "groq":
        return GroqLLM()
    elif LLM_PROVIDER == "gemini":
        return GeminiLLM()
    else:
        raise ValueError(f"unknown LLM_PROVIDER: {LLM_PROVIDER}")


class GroqLLM:
    

    def __init__(self):
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.provider_name = "groq"

    def generate(self, system_prompt, user_message, temperature=0.1, max_tokens=1024):
        """sending a prompt to LLM and returning text response"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


class GeminiLLM:
    """google gemini — drop-in replacement for groq"""

    def __init__(self):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.provider_name = "gemini"

    def generate(self, system_prompt, user_message, temperature=0.1, max_tokens=1024):
        """send a prompt to gemini and return the text response"""
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=f"{system_prompt}\n\n{user_message}",
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()