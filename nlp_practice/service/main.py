import uvicorn
from fastapi import FastAPI
from llm.service.api.v1 import openai
from llm.service.core.settings.config import Settings

app = FastAPI(
    title=Settings().app_name,
    version="0.0.0",
)


@app.get("/")
def index() -> str:
    return "LLM Service Entry"


app.include_router(openai.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
