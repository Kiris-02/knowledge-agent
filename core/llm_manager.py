import os
from typing import Generator, List, Dict, Any, Optional
from openai import OpenAI
try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

class LLMManager:
    def __init__(self, deepseek_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        self.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        
        self.deepseek_client = None
        if self.deepseek_api_key:
            self.deepseek_client = OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
            
        self.gemini_client = None
        if self.gemini_api_key and HAS_GOOGLE_GENAI:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)

    def set_keys(self, deepseek_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        if deepseek_api_key:
            self.deepseek_api_key = deepseek_api_key
            self.deepseek_client = OpenAI(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com"
            )
        if gemini_api_key and HAS_GOOGLE_GENAI:
            self.gemini_api_key = gemini_api_key
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)

    def stream_chat(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.5
    ) -> Generator[str, None, None]:
        """Unified streaming generator across DeepSeek and Gemini."""
        provider = provider.lower()
        
        if provider == "deepseek":
            if not self.deepseek_client:
                raise ValueError("Chưa cấu hình DEEPSEEK_API_KEY. Vui lòng nhập API Key trong thanh bên (Sidebar) hoặc file .env")
            
            api_messages = [{"role": "system", "content": system_prompt}]
            api_messages.extend(messages)
            
            response = self.deepseek_client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif provider == "gemini":
            if not self.gemini_client:
                raise ValueError("Chưa cấu hình GEMINI_API_KEY. Vui lòng nhập API Key trong thanh bên (Sidebar) hoặc file .env")
            
            # Combine history into Gemini format
            prompt_parts = [f"System Instruction:\n{system_prompt}\n"]
            for m in messages:
                role = "User" if m["role"] == "user" else "Assistant"
                prompt_parts.append(f"{role}: {m['content']}")
            
            full_prompt = "\n\n".join(prompt_parts)
            
            response = self.gemini_client.models.generate_content_stream(
                model=model,
                contents=full_prompt,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        else:
            raise ValueError(f"Provider '{provider}' không được hỗ trợ. Chọn 'deepseek' hoặc 'gemini'.")
