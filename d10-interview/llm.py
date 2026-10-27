"""OpenAI 호출은 이 파일 한 곳에서만 합니다."""

import json
import os

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

DEFAULT_MODEL = "gpt-6-luna"


class LLMError(Exception):
    """화면에 그대로 보여 줄 한국어 한 줄 에러."""


def ask_json(instructions: str, user_input: str, schema: dict, name: str) -> dict:
    """Responses API 를 strict JSON 스키마로 불러 dict 로 돌려줍니다."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise LLMError("OpenAI API 키가 없습니다. .env 의 OPENAI_API_KEY 를 확인해 주세요.")
    model = os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_MODEL

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_input,
            text={
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )
    except AuthenticationError:
        raise LLMError("API 키가 올바르지 않습니다. .env 의 OPENAI_API_KEY 를 확인해 주세요.") from None
    except RateLimitError as e:
        if getattr(e, "code", None) == "insufficient_quota":
            raise LLMError("OpenAI 크레딧이 부족합니다. 결제 · 사용 한도를 확인해 주세요.") from None
        raise LLMError("요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.") from None
    except APIConnectionError:
        raise LLMError("OpenAI 서버에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요.") from None
    except APIError:
        raise LLMError("AI 응답을 받지 못했습니다. 잠시 후 다시 시도해 주세요.") from None

    try:
        return json.loads(response.output_text)
    except (json.JSONDecodeError, TypeError):
        raise LLMError("AI 응답을 읽지 못했습니다. 다시 시도해 주세요.") from None
