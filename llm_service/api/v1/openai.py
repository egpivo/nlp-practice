import os

import openai
from dotenv import load_dotenv
from fastapi import APIRouter

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")
router = APIRouter(
    prefix="/openai",
    tags=["prompt"],
)


@router.get("/completions")
def completions(
    prompt: str,
    model: str = "gpt-3.5-turbo-instruct",
    temperature: float = 0,
    max_tokens: int = 150,
    top_p: int = 1,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
) -> dict:
    return openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )


@router.post("/chat/completions")
def chat_completions(
    messages: list[dict[str, str]],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.9,
) -> dict:
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )


@router.get("/embeddings")
def embeddings(input: str, model: str = "text-embedding-ada-002") -> dict:
    return openai.Embedding.create(
        model=model,
        input=input,
    )
