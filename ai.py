"""The one AI capability: report + patient's own words  ->  structured plain-language explanation."""
import time
from typing import Optional

from schema import Explanation
import re

from prompts import FOLLOWUP_SYSTEM, SYSTEM_INSTRUCTION, build_followup_message, build_user_message
from samples import SAMPLES, demo_explanation
from verify import check_completeness, verify

DEFAULT_MODEL = "gemini-3.8-flash"
SUPPORTED_MIME = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


class AnalysisError(Exception):
    """Raised with a message that is safe and friendly to show the user."""


def analyze(
    concern: str,
    api_key: Optional[str],
    file_bytes: Optional[bytes] = None,
    file_name: str = "",
    report_text: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> tuple[Explanation, bool]:
    """Returns (explanation, is_demo). Demo mode is used only when there is no API key
    AND the input is one of the built-in sample reports."""
    if not api_key:
        kind = SAMPLES.get((report_text or "").strip())
        if kind:
            return demo_explanation(kind), True
        raise AnalysisError(
            "No Gemini API key is configured, so only the built-in sample reports can be explained. "
            "Add GEMINI_API_KEY in the app settings to analyse your own file."
        )

    if not file_bytes and not (report_text and report_text.strip()):
        raise AnalysisError("Please add a report first (PDF, photo, or pasted text).")

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise AnalysisError("The AI library is not installed. Run: pip install -r requirements.txt") from e

    parts = []
    if file_bytes:
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        mime = SUPPORTED_MIME.get(ext)
        if not mime:
            raise AnalysisError("Unsupported file type. Please upload a PDF, PNG, JPG or WEBP.")
        parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime))
    if report_text and report_text.strip():
        parts.append("REPORT TEXT:\n" + report_text.strip())
    parts.append(build_user_message(concern))

    # Hard time limit per attempt so the app never hangs for minutes.
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=80_000))  # 80 s: long reports have many findings
    candidates = []
    for m in [model, "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]:
        if m and m not in candidates:
            candidates.append(m)

    def _busy(low):
        return any(k in low for k in ("503", "unavailable", "overloaded", "high demand", "429",
                                      "resource_exhausted"))

    def _config(fast: bool):
        kwargs = dict(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=Explanation,
            temperature=0.2,
            max_output_tokens=20000,
        )
        if fast:
            # Little/no "thinking": reading a report into a fixed JSON shape does not need it.
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kwargs)

    last_error = None
    errors = []
    result = None
    started = time.monotonic()
    MAX_TOTAL = 170  # seconds: never keep the user waiting longer than this overall
    for m in candidates:
        fast = True  # each model starts with the quick setting
        if errors and time.monotonic() - started > MAX_TOTAL:
            break
        attempt = 0
        tries = 0
        while tries < 3 and attempt < 2:  # at most one busy-retry per model (+1 for the thinking fallback)
            tries += 1
            try:
                resp = client.models.generate_content(model=m, contents=parts, config=_config(fast))
                parsed = getattr(resp, "parsed", None)
                result = parsed if isinstance(parsed, Explanation) \
                    else Explanation.model_validate_json(resp.text)
                break
            except Exception as e:  # noqa: BLE001
                last_error = e
                errors.append((m, type(e).__name__ + ": " + str(e)[:160]))
                low = str(e).lower()
                if "thinking" in low and fast:
                    fast = False  # this model rejects the thinking setting: retry without it (free retry)
                    continue
                if "404" in low or "not_found" in low or "no longer available" in low:
                    break  # model gone: next model
                if (_busy(low) or "timed out" in low or "timeout" in low) and attempt == 0 \
                        and time.monotonic() - started < MAX_TOTAL:
                    attempt += 1
                    time.sleep(1.5)
                    continue
                break
        if result is not None:
            break

    if result is None:
        # Prefer the most useful error: a "model not found" from one candidate must not hide a busy/key/timeout problem on another.
        def _rank(msg):
            l = msg.lower()
            if any(k in l for k in ("api key", "api_key_invalid", "401", "403", "permission")): return 0
            if any(k in l for k in ("503", "unavailable", "high demand", "429", "quota", "resource_exhausted")): return 1
            if any(k in l for k in ("timed out", "timeout", "deadline")): return 2
            if any(k in l for k in ("404", "not_found")): return 4
            return 3
        first_err = min((e_[1] for e_ in errors), key=_rank) if errors else "unknown error"
        shown = " | ".join(f"{m}: {msg}" for m, msg in errors[:4]) or first_err
        if api_key:
            shown = shown.replace(api_key, "***")
        detail = f" (technical detail: {shown[:700]})"
        low = first_err.lower()
        e = last_error
        if "api key" in low or "api_key_invalid" in low or "401" in low or "403" in low or "permission" in low:
            raise AnalysisError("The AI service rejected the API key. Check GEMINI_API_KEY in Secrets." + detail) from e
        if "503" in low or "unavailable" in low or "high demand" in low or "429" in low or "quota" in low or "resource_exhausted" in low:
            raise AnalysisError("Google's AI service is very busy right now. Please wait a minute and press Explain my report again." + detail) from e
        if "timed out" in low or "timeout" in low or "deadline" in low:
            raise AnalysisError("The AI took too long to answer. Please press Explain my report again." + detail) from e
        if "404" in low or "not_found" in low:
            raise AnalysisError("The AI model name is not available. Set GEMINI_MODEL in Secrets to a current model." + detail) from e
        raise AnalysisError("Something went wrong while reading the report. Please try again, or try a clearer file." + detail) from e

    if not result.findings and result.is_medical_report:
        raise AnalysisError("The AI could not pick out the individual results in this document. Please press "
                            "Explain my report again, or try a clearer file.")
    check_completeness(result)
    changed = verify(result)
    if changed:
        result.limits_note = (result.limits_note + " Some statuses were re-checked automatically against the "
                              "ranges printed in your report.").strip()
    return result, False


def list_models(api_key: str) -> list[str]:
    """Names of models this key can use for generating content (used by the sidebar check)."""
    from google import genai
    client = genai.Client(api_key=api_key)
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            names.append((m.name or "").replace("models/", ""))
    return sorted(set(n for n in names if n))


# ---------------------------------------------------------------------------
# Follow-up questions: bounded, report-only, nothing saved
# ---------------------------------------------------------------------------
MAX_FOLLOWUPS = 5
MAX_QUESTION_CHARS = 400

# Questions we never send to the AI: they ask for medicine/dose/treatment decisions.
_BLOCKED = re.compile(
    r"\b(dose|dosage|how (much|many) (mg|tablets?|pills?)|should i (take|stop|start|skip|increase|reduce)|"
    r"prescri\w+|which (medicine|medication|drug|tablet|supplement)|what (medicine|medication|drug|tablet|supplement)|"
    r"stop (taking|my)|replace .{0,20}medic\w+|do i have (cancer|diabetes|a tumou?r)|diagnos\w+ me)\b", re.I)

BLOCKED_REPLY = ("I can explain what your results mean, but I can't advise on medicines, doses or treatment, "
                 "and I can't diagnose. Please take this question to your doctor or pharmacist. "
                 "I can help you word it if you like.")


def followup_context(result) -> str:
    """Compact text version of the explanation. This is the ONLY thing follow-ups can draw on."""
    lines = [f"Headline: {result.headline}", f"Summary: {result.summary}", "Results:"]
    for f in result.findings:
        lines.append(f"- {f.name}: {f.value_text} {f.unit} | printed range: {f.reference_text or 'not printed'} | "
                     f"status: {f.status} | from report: {f.source_quote} | explained: {f.plain_meaning[:300]}")
    return "\n".join(lines)[:24000]


def answer_followup(question: str, result, history, api_key, model: str = DEFAULT_MODEL) -> str:
    question = (question or "").strip()
    if not question:
        raise AnalysisError("Please type a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise AnalysisError(f"Please keep the question under {MAX_QUESTION_CHARS} characters.")
    if len(history) >= MAX_FOLLOWUPS:
        raise AnalysisError(f"You have used all {MAX_FOLLOWUPS} follow-up questions for this report.")
    if _BLOCKED.search(question):
        return BLOCKED_REPLY
    if not api_key:
        raise AnalysisError("Follow-up questions need the live AI, which is not switched on in demo mode.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        raise AnalysisError("The AI library is not installed.") from e

    contents = build_followup_message(followup_context(result), question, history)
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))
    candidates = []
    for m in [model, "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]:
        if m and m not in candidates:
            candidates.append(m)
    started, errors = time.monotonic(), []
    for m in candidates:
        if errors and time.monotonic() - started > 70:
            break
        fast = True
        for attempt in range(3):
            try:
                kw = dict(system_instruction=FOLLOWUP_SYSTEM, temperature=0.2, max_output_tokens=900)
                if fast:
                    kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                resp = client.models.generate_content(model=m, contents=contents,
                                                      config=types.GenerateContentConfig(**kw))
                text = (getattr(resp, "text", None) or "").strip()
                if not text:
                    raise ValueError("empty answer")
                return text
            except Exception as e:  # noqa: BLE001
                low = str(e).lower()
                errors.append(str(e)[:120])
                if "thinking" in low and fast:
                    fast = False
                    continue
                if ("503" in low or "unavailable" in low or "429" in low or "timed out" in low) and attempt < 2 \
                        and time.monotonic() - started < 70:
                    time.sleep(1.2)
                    continue
                break
    shown = " | ".join(errors[:2]).replace(api_key, "***")
    raise AnalysisError("I couldn't answer that just now. Please try again in a moment." + f" (detail: {shown})")
