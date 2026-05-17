from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from supabase_client import get_active_pipeline_history, save_agent_storyboard


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    """Application settings loaded from the local environment."""

    environment: str = Field(default="development", alias="ENVIRONMENT")
    port: int = Field(default=8000, alias="PORT")
    tokenrouter_api_key: str = Field(alias="TOKENROUTER_API_KEY")
    tokenrouter_base_url: str = Field(
        default="https://api.tokenrouter.com/v1",
        alias="TOKENROUTER_BASE_URL",
    )
    tokenrouter_analyst_model: str = Field(
        default="qwen/qwen3.5-122b-a10b",
        alias="TOKENROUTER_ANALYST_MODEL",
    )
    tokenrouter_director_model: str = Field(
        default="z-ai/glm-5.1",
        alias="TOKENROUTER_DIRECTOR_MODEL",
    )
    enable_local_sentiment_model: bool = Field(
        default=False,
        alias="ENABLE_LOCAL_SENTIMENT_MODEL",
    )
    market_data_api_key: str = Field(alias="MARKET_DATA_API_KEY")
    news_research_api_key: str = Field(alias="NEWS_RESEARCH_API_KEY")
    huggingface_token: str | None = Field(default=None, alias="HUGGINGFACE_TOKEN")
    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")
    replicate_api_key: str | None = Field(default=None, alias="REPLICATE_API_KEY")
    runway_api_key: str | None = Field(default=None, alias="RUNWAY_API_KEY")
    social_distribution_api_key: str | None = Field(
        default=None,
        alias="SOCIAL_DISTRIBUTION_API_KEY",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


@lru_cache(maxsize=1)
def get_tokenrouter_client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.tokenrouter_api_key,
        base_url=settings.tokenrouter_base_url,
    )


app = FastAPI(
    title="Multi-Agent Content Storyboard SaaS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def _score_topic_sentiment(topic: str) -> float:
    positive_terms = {
        "accelerate",
        "beat",
        "breakout",
        "bullish",
        "growth",
        "improve",
        "momentum",
        "rally",
        "record",
        "resistance",
        "surge",
        "upside",
    }
    negative_terms = {
        "bearish",
        "breakdown",
        "compress",
        "concern",
        "decline",
        "disappoint",
        "fall",
        "fear",
        "risk",
        "selloff",
        "slow",
        "weak",
    }
    tokens = {
        token.strip(".,!?;:()[]{}\"'").lower()
        for token in topic.split()
        if token.strip(".,!?;:()[]{}\"'")
    }
    positive_score = len(tokens & positive_terms)
    negative_score = len(tokens & negative_terms)
    if positive_score == negative_score:
        return 0.0
    return max(-1.0, min(1.0, (positive_score - negative_score) / 6))


def _normalize_sentiment(raw_result: dict[str, Any]) -> float:
    label = str(raw_result.get("label", "")).strip().lower()
    score = float(raw_result.get("score", 0.0))

    if "negative" in label:
        return -score
    if "positive" in label:
        return score
    return 0.0


def _map_market_bias(sentiment_score: float) -> str:
    if sentiment_score > 0.15:
        return "Long"
    if sentiment_score < -0.15:
        return "Short"
    return "Neutral"


def _extract_message_content(response: Any) -> str:
    if not response.choices:
        raise RuntimeError("Model response did not include any choices.")

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Model response returned empty message content.")

    return content.strip()


def _parse_json_object(raw_content: str, agent_name: str) -> dict[str, Any]:
    cleaned_content = raw_content.strip()
    if cleaned_content.startswith("```"):
        cleaned_content = cleaned_content.removeprefix("```json").removeprefix("```")
        cleaned_content = cleaned_content.removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned_content)
    except json.JSONDecodeError as exc:
        logger.exception("%s returned malformed JSON: %s", agent_name, cleaned_content)
        raise RuntimeError(f"{agent_name} returned malformed JSON.") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"{agent_name} did not return a JSON object.")

    return parsed


async def _run_market_analyst_agent(
    *,
    topic: str,
    sentiment_score: float,
    market_bias: str,
    settings: AppSettings,
    client: AsyncOpenAI,
) -> dict[str, Any]:
    logger.info("Running Market Analyst Agent for topic=%s.", topic)
    response = await client.chat.completions.create(
        model=settings.tokenrouter_analyst_model,
        max_tokens=1500,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are The Market Analyst Agent. Return only one strict JSON "
                    "object with no markdown. Build a concise investment strategy "
                    "positioning hypothesis."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "topic": topic,
                        "local_sentiment_score": sentiment_score,
                        "computed_market_bias": market_bias,
                        "simulated_ingestion_context": {
                            "market_data_api_key_configured": bool(
                                settings.market_data_api_key
                            ),
                            "news_research_api_key_configured": bool(
                                settings.news_research_api_key
                            ),
                            "instruction": (
                                "Simulate a live ingestion loop that would query the "
                                "MARKET_DATA_API_KEY and NEWS_RESEARCH_API_KEY backed "
                                "services for price action, catalysts, volatility, and "
                                "breaking-news context. Do not fabricate exact prices."
                            ),
                        },
                        "required_json_keys": [
                            "topic_headline",
                            "sentiment_interpretation",
                            "market_bias",
                            "positioning_hypothesis",
                            "key_catalysts",
                            "risk_factors",
                        ],
                    }
                ),
            },
        ],
    )
    return _parse_json_object(
        _extract_message_content(response),
        "Market Analyst Agent",
    )


async def _run_creative_video_director_agent(
    *,
    analyst_packet: dict[str, Any],
    settings: AppSettings,
    client: AsyncOpenAI,
) -> dict[str, Any]:
    logger.info("Running Creative Video Director Agent.")
    response = await client.chat.completions.create(
        model=settings.tokenrouter_director_model,
        max_tokens=2000,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are The Creative Video Director Agent. Return only one strict "
                    "JSON object with no markdown. Convert market analysis into a "
                    "15-second viral vertical finance storyboard. Keep every field "
                    "concise enough to fit in one complete JSON response."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "analyst_packet": analyst_packet,
                        "required_json_keys": [
                            "video_title",
                            "audio_script",
                            "visual_prompts",
                        ],
                        "constraints": {
                            "format": "vertical 9:16",
                            "duration_seconds": 15,
                            "style": "clear, high-retention, finance-native",
                            "visual_prompts": (
                                "Return a single compact string with three scene prompts, "
                                "not an array, and keep it under 900 characters."
                            ),
                            "audio_script": "Keep narration under 65 words.",
                        },
                    }
                ),
            },
        ],
    )
    return _parse_json_object(
        _extract_message_content(response),
        "Creative Video Director Agent",
    )


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting application lifecycle initialization.")
    settings = get_settings()
    logger.info("Runtime environment=%s port=%s.", settings.environment, settings.port)

    app.state.sentiment_pipeline = None
    if not settings.enable_local_sentiment_model:
        logger.info("Using lightweight sentiment scorer for deployment runtime.")
        return

    try:
        from transformers import pipeline

        app.state.sentiment_pipeline = await asyncio.to_thread(
            lambda: pipeline(
                "sentiment-analysis",
                model=SENTIMENT_MODEL_NAME,
                tokenizer=SENTIMENT_MODEL_NAME,
                device=-1,
            )
        )
        logger.info("Loaded local CPU sentiment model: %s.", SENTIMENT_MODEL_NAME)
    except Exception:
        logger.exception("Failed to preload local Hugging Face sentiment classifier.")
        raise


@app.get("/")
async def dashboard(request: Request):
    try:
        history = await get_active_pipeline_history()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "history": history,
            },
        )
    except Exception:
        logger.exception("Failed to render content storyboard dashboard.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load dashboard history.",
        )


@app.post("/api/v1/generate-content", status_code=status.HTTP_201_CREATED)
async def generate_content(request: Request, topic: str = Form(...)) -> JSONResponse:
    try:
        normalized_topic = topic.strip()
        if not normalized_topic:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Topic is required.",
            )

        settings = get_settings()
        tokenrouter_client = get_tokenrouter_client()

        logger.info("Generating multi-agent content for topic=%s.", normalized_topic)
        sentiment_pipeline = getattr(request.app.state, "sentiment_pipeline", None)
        if sentiment_pipeline is None:
            sentiment_score = _score_topic_sentiment(normalized_topic)
        else:
            sentiment_results = await asyncio.to_thread(
                sentiment_pipeline,
                normalized_topic,
            )
            if not sentiment_results:
                raise RuntimeError("Local sentiment classifier returned no result.")
            sentiment_score = _normalize_sentiment(sentiment_results[0])

        market_bias = _map_market_bias(sentiment_score)
        logger.info(
            "Local sentiment complete for topic=%s score=%s market_bias=%s.",
            normalized_topic,
            sentiment_score,
            market_bias,
        )

        analyst_packet = await _run_market_analyst_agent(
            topic=normalized_topic,
            sentiment_score=sentiment_score,
            market_bias=market_bias,
            settings=settings,
            client=tokenrouter_client,
        )
        director_packet = await _run_creative_video_director_agent(
            analyst_packet=analyst_packet,
            settings=settings,
            client=tokenrouter_client,
        )

        payload = {
            "topic_headline": str(
                analyst_packet.get("topic_headline", normalized_topic)
            ),
            "sentiment_score": sentiment_score,
            "market_bias": str(analyst_packet.get("market_bias", market_bias)),
            "video_title": str(director_packet["video_title"]),
            "audio_script": str(director_packet["audio_script"]),
            "visual_prompts": json.dumps(
                director_packet["visual_prompts"],
                ensure_ascii=False,
            )
            if not isinstance(director_packet["visual_prompts"], str)
            else director_packet["visual_prompts"],
            "status": "Pending Video Production",
        }

        saved_row = await save_agent_storyboard(payload)

        # ==============================================================================
        # FUTURE PRODUCTION AUTOMATION STEP (Paid Rendering & Social Distribution Pipeline)
        # ==============================================================================
        # async with httpx.AsyncClient(timeout=120) as media_client:
        #     voiceover_response = await media_client.post(
        #         "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        #         headers={"xi-api-key": settings.elevenlabs_api_key},
        #         json={
        #             "text": payload["audio_script"],
        #             "model_id": "eleven_multilingual_v2",
        #         },
        #     )
        #     voiceover_response.raise_for_status()
        #     media_audio_url = upload_audio_bytes_to_object_storage(
        #         voiceover_response.content
        #     )
        #
        #     image_response = await media_client.post(
        #         "https://api.replicate.com/v1/predictions",
        #         headers={"Authorization": f"Bearer {settings.replicate_api_key}"},
        #         json={
        #             "version": "flux-production-model-version",
        #             "input": {"prompt": payload["visual_prompts"], "aspect_ratio": "9:16"},
        #         },
        #     )
        #     image_response.raise_for_status()
        #     background_asset_urls = await poll_replicate_prediction_until_complete(
        #         image_response.json()
        #     )
        #
        #     vertical_mp4_path = await asyncio.to_thread(
        #         render_vertical_storyboard_with_moviepy,
        #         audio_url=media_audio_url,
        #         background_asset_urls=background_asset_urls,
        #         title_overlay=payload["video_title"],
        #         output_resolution=(1080, 1920),
        #     )
        #
        #     publish_response = await media_client.post(
        #         "https://social-distribution.example.com/webhooks/publish",
        #         headers={"Authorization": f"Bearer {settings.social_distribution_api_key}"},
        #         json={
        #             "video_path": str(vertical_mp4_path),
        #             "caption": payload["video_title"],
        #             "channels": ["instagram_reels", "youtube_shorts", "tiktok"],
        #         },
        #     )
        #     publish_response.raise_for_status()

        logger.info(
            "Multi-agent content generation succeeded row_id=%s.",
            saved_row.get("id"),
        )
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "status": "created",
                "data": saved_row,
                "agent_outputs": {
                    "market_analyst": analyst_packet,
                    "creative_video_director": director_packet,
                },
            },
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to generate multi-agent content.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content generation pipeline failed.",
        )


if __name__ == "__main__":
    import uvicorn

    runtime_settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=runtime_settings.port,
        reload=runtime_settings.environment == "development",
    )
