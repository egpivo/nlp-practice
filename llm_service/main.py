import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_service.api.v1 import openai
from llm_service.core.settings.config import Settings

app = FastAPI(
    title=Settings().app_name,
    version="0.1.0",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> str:
    return "Welcome to LLM Services"


app.include_router(openai.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
