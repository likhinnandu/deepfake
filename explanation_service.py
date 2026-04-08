import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(override=False)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def _default_summary(result_percent: int, media_type: str) -> str:
    kind = "audio" if media_type == "audio" else "video"
    if result_percent >= 75:
        return (
            f"The {kind} appears highly suspicious. Multiple manipulation signals "
            "were detected and the probability of tampering is high."
        )
    if result_percent >= 50:
        return (
            f"The {kind} has mixed signals. Some manipulation indicators were detected, "
            "but they are not strong enough for a definitive conclusion."
        )
    return (
        f"The {kind} appears mostly authentic. The analysis found only limited "
        "or weak manipulation indicators."
    )


def _local_fallback(
    media_name: str,
    media_type: str,
    result_percent: int,
    reasoning_data: Dict[str, Any],
    reason_code: str = "default",
    reason_detail: Optional[str] = None,
) -> Dict[str, Any]:
    summary = _default_summary(result_percent, media_type)

    reason_messages = {
        "missing_api_key": {
            "mode": "Using local explanation fallback because Gemini API key is missing.",
            "search": "Google search verification is not enabled. Add GEMINI_API_KEY (or GOOGLE_API_KEY) to .env and restart the app.",
        },
        "missing_sdk": {
            "mode": "Using local explanation fallback because Gemini SDK is unavailable.",
            "search": "Google search verification is not enabled because google-genai is unavailable in the current environment.",
        },
        "invalid_api_key": {
            "mode": "Using local explanation fallback because Gemini rejected the configured API key.",
            "search": "Google search verification was skipped because the configured Gemini key is invalid, revoked, or leaked. Create a new key in Google AI Studio, update .env, and restart the app.",
        },
        "quota_exceeded": {
            "mode": "Using local explanation fallback because Gemini quota/rate limits were reached.",
            "search": "Google search verification was skipped because Gemini quota is exhausted or requests are rate-limited. Retry later or increase quota.",
        },
        "invalid_response": {
            "mode": "Using local explanation fallback because Gemini returned an invalid response format.",
            "search": "Google search verification could not be interpreted reliably from Gemini output, so local forensic explanation is shown.",
        },
        "default": {
            "mode": "Using local explanation fallback because Gemini is currently unavailable.",
            "search": "Google search verification is unavailable right now, so this explanation is based on local forensic signals.",
        },
    }

    selected = reason_messages.get(reason_code, reason_messages["default"])
    search_summary = selected["search"]

    fallback_points = [
        {
            "icon": "fa-robot",
            "title": "AI Explanation Mode",
            "detail": selected["mode"],
        },
        {
            "icon": "fa-magnifying-glass",
            "title": "Google Search Check",
            "detail": search_summary,
        },
    ]

    if reason_detail:
        detail_text = str(reason_detail)
        if len(detail_text) > 320:
            detail_text = f"{detail_text[:317]}..."
        fallback_points.append(
            {
                "icon": "fa-circle-info",
                "title": "Gemini Status",
                "detail": detail_text,
            }
        )

    return {
        "summary": summary,
        "search_summary": search_summary,
        "search_found_online": None,
        "reasoning_points": fallback_points,
        "source": f"local:{reason_code}",
    }


def generate_media_explanation(
    media_name: str,
    media_type: str,
    result_percent: int,
    reasoning_data: Dict[str, Any],
    audio_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a user-friendly explanation and a web-search-based traceability note.

    If Gemini + Google Search is available, use it. Otherwise fallback to a
    deterministic local explanation.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return _local_fallback(
            media_name,
            media_type,
            result_percent,
            reasoning_data,
            reason_code="missing_api_key",
        )
    if genai is None:
        return _local_fallback(
            media_name,
            media_type,
            result_percent,
            reasoning_data,
            reason_code="missing_sdk",
        )

    compact_metrics = {
        "total_frames": reasoning_data.get("total_frames", 0),
        "frames_processed": reasoning_data.get("frames_processed", 0),
        "faces_detected": reasoning_data.get("faces_detected", 0),
        "deepfake_frames": reasoning_data.get("deepfake_frames", 0),
        "avg_similarity": reasoning_data.get("avg_similarity", 0),
        "face_detection_rate": reasoning_data.get("face_detection_rate", 0),
        "execution_time": reasoning_data.get("execution_time", 0),
        "video_resolution": reasoning_data.get("video_resolution", "N/A"),
        "video_fps": reasoning_data.get("video_fps", 0),
    }

    prompt_payload = {
        "media_name": media_name,
        "media_type": media_type,
        "score_percent": result_percent,
        "metrics": compact_metrics,
        "audio_results": audio_results or {},
    }

    prompt = (
        "You are an AI forensic assistant for deepfake detection. "
        "Use the provided metrics and perform a Google search for the media name/title to check "
        "whether related versions or references appear online. "
        "If search confidence is weak, say that clearly. "
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        "  \"summary\": string,\n"
        "  \"search_summary\": string,\n"
        "  \"search_found_online\": boolean|null,\n"
        "  \"reasoning_points\": [\n"
        "    {\"icon\": string, \"title\": string, \"detail\": string}\n"
        "  ]\n"
        "}\n"
        "Keep summary under 90 words. reasoning_points max 3.\n"
        f"Input JSON:\n{json.dumps(prompt_payload)}"
    )

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    try:
        client = genai.Client(api_key=api_key)

        config = None
        if types is not None:
            config = types.GenerateContentConfig(
                temperature=0.2,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        raw_text = getattr(response, "text", "")
        parsed = _extract_json_object(raw_text)
        if not parsed:
            return _local_fallback(
                media_name,
                media_type,
                result_percent,
                reasoning_data,
                reason_code="invalid_response",
                reason_detail="Gemini response did not contain valid JSON in the expected schema.",
            )

        default_payload = _local_fallback(
            media_name,
            media_type,
            result_percent,
            reasoning_data,
        )

        summary = str(parsed.get("summary") or default_payload["summary"])
        search_summary = str(parsed.get("search_summary") or default_payload["search_summary"])
        search_found_online = parsed.get("search_found_online")
        reasoning_points = parsed.get("reasoning_points") or []

        safe_points: List[Dict[str, str]] = []
        for point in reasoning_points[:3]:
            if not isinstance(point, dict):
                continue
            safe_points.append(
                {
                    "icon": str(point.get("icon", "fa-circle-info")),
                    "title": str(point.get("title", "AI Insight")),
                    "detail": str(point.get("detail", "")),
                }
            )

        if not safe_points:
            local_fallback = _local_fallback(
                media_name,
                media_type,
                result_percent,
                reasoning_data,
                reason_code="invalid_response",
            )
            safe_points = local_fallback["reasoning_points"]

        return {
            "summary": summary,
            "search_summary": search_summary,
            "search_found_online": search_found_online,
            "reasoning_points": safe_points,
            "source": "gemini",
        }
    except Exception as exc:
        error_text = str(exc)
        lowered = error_text.lower()

        reason_code = "default"
        if (
            "permission_denied" in lowered
            or "api key not valid" in lowered
            or "invalid api key" in lowered
            or "leaked" in lowered
            or "403" in lowered
            or "api_key_invalid" in lowered
        ):
            reason_code = "invalid_api_key"
        elif (
            "quota" in lowered
            or "rate limit" in lowered
            or "resource_exhausted" in lowered
            or "429" in lowered
        ):
            reason_code = "quota_exceeded"

        return _local_fallback(
            media_name,
            media_type,
            result_percent,
            reasoning_data,
            reason_code=reason_code,
            reason_detail=f"Gemini explanation failed: {error_text}",
        )
