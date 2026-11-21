from pathlib import Path
from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.pipeline import process_text
from app.schemas import DocumentOutput

app = FastAPI(
    title="Financial Document Extractor API",
    description="Сервис структурированного извлечения реквизитов из счетов, актов и КП (KZ/RU/EN)",
    version="1.0.0",
)
STATIC_INDEX = Path(__file__).parent / "static" / "index.html"

class ParseRequest(BaseModel):
    text: str = Field(..., description="Сырой текст документа после OCR или копирования из PDF")
    use_mock: bool = Field(
        default=False,
        description="Принудительно использовать локальный MockProvider без обращения к Gemini API",
    )


class HealthResponse(BaseModel):
    status: str
    mode: str
    model: str

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index_page() -> HTMLResponse:
    return HTMLResponse(content=STATIC_INDEX.read_text(encoding="utf-8"))

@app.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        mode="mock" if settings.is_mock_mode else "gemini",
        model=settings.gemini_model,
    )


@app.post("/parse", response_model=DocumentOutput)
def parse_document(
    payload: ParseRequest,
    settings: Settings = Depends(get_settings),
) -> DocumentOutput:
    return process_text(
        raw_text=payload.text,
        use_mock=payload.use_mock,
        settings=settings,
    )