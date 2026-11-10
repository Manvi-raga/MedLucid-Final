"""Run:  pip install pytest   then   python -m pytest tests -q   (from the app folder)."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from schema import Explanation, Finding, Term
import verify as V
import ui, visuals, pdf_export, ai
from samples import demo_explanation, SAMPLE_REPORT, SAMPLE_REPORT_NORMAL


def F(name="Haemoglobin", num=13.0, lo=12.0, hi=15.0, status="normal", quote=None, **kw):
    return Finding(name=name, value_text=str(num), unit="g/dL", numeric_value=num, ref_low=lo, ref_high=hi,
                   reference_text=f"{lo}-{hi}", status=status, plain_meaning="meaning",
                   source_quote=quote if quote is not None else f"{name} {num} {lo}-{hi}", **kw)


def E(findings, **kw):
    d = dict(is_medical_report=True, headline="h", summary="s", findings=findings, concern_answer="c",
             questions_for_doctor=["q"], terms_explained=[Term(term="t", meaning="m")], limits_note="l")
    d.update(kw)
    return Explanation(**d)


# ---------- verify ----------
def test_status_upgrade_outside_range():
    r = E([F(num=10.0, status="normal")]); assert V.verify(r) == 1
    assert r.findings[0].status in ("borderline", "attention")

def test_status_downgrade_inside():
    r = E([F(num=13.0, status="attention")]); V.verify(r); assert r.findings[0].status == "borderline"

def test_boundary_inclusive():
    r = E([F(num=15.0, status="normal")]); assert V.verify(r) == 0

def test_swapped_range_does_not_flag_everything():
    r = E([F(num=13.0, lo=15.0, hi=12.0, status="normal")]); V.verify(r)
    assert r.findings[0].status == "normal"

def test_one_sided_ranges():
    r = E([F("LDL", 142, None, 100, "normal"), F("HDL", 30, 40, None, "normal")]); V.verify(r)
    assert all(f.status != "normal" for f in r.findings)

def test_zero_bounds_no_crash():
    r = E([F(num=-1, lo=0, hi=0, status="normal"), F(num=5, lo=0, hi=0, status="normal")]); V.verify(r)

def test_misread_note():
    r = E([F(num=13.0, quote="Haemoglobin 31.0 12-15")]); V.verify(r); assert r.findings[0].check_note

def test_ai_written_note_is_wiped():
    r = E([F(check_note="inject")]); V.verify(r); assert r.findings[0].check_note is None

def test_qualitative_finding_skipped():
    f = Finding(name="Urine protein", value_text="Negative", status="normal", plain_meaning="m", source_quote="Negative")
    V.verify(E([f]))

def test_completeness():
    r = E([F()], tests_seen=5); V.check_completeness(r); assert r.completeness_note
    r = E([F()], tests_seen=1); V.check_completeness(r); assert r.completeness_note is None

def test_zero_findings_is_flagged():
    r = E([], tests_seen=0); V.check_completeness(r)
    assert r.completeness_note, "an explanation with no results must never look complete"


# ---------- visuals / ui ----------
@pytest.mark.parametrize("name,expect", [
    ("Haemoglobin", "haemoglobin"), ("HbA1c", "sugar_a1c"), ("Fasting Blood Glucose", "sugar"),
    ("LDL Cholesterol", "cholesterol"), ("HDL Cholesterol", None), ("Total Cholesterol", "cholesterol"),
    ("VLDL Cholesterol", None), ("Mean Corpuscular Haemoglobin (MCH)", None),
    ("Mean Corpuscular Hb Conc (MCHC)", None), ("Urine Glucose", None), ("Urine Sugar", None),
    ("Non-HDL Cholesterol", None), ("Cholesterol/HDL ratio", None), ("TSH", None), ("", None),
])
def test_kind_for(name, expect):
    assert visuals.kind_for(name) == expect

def test_all_pictures_render():
    for n in ("Haemoglobin", "HbA1c", "Glucose", "LDL"):
        for v in (5, 13, 99):
            h = visuals.picture_html(F(n, v, 12, 15)); assert "<svg" in h

def test_html_escape_everywhere():
    evil = '<script>alert(1)</script>'
    f = F(name=evil, quote=evil); f.plain_meaning = evil; f.unit = evil
    h = ui.finding_html(f); assert "<script>" not in h
    assert "<script>" not in ui.questions_html([evil])

@pytest.mark.parametrize("v,lo,hi", [(5, 1, 10), (0, 0, 0), (10, 10, 10), (-3, None, 4), (1e9, 1, 2), (3, None, None),
                                      (5, 7, None), (0.0001, 0.0002, 0.0003), (float("inf"), 1, 2)])
def test_range_bar_never_crashes(v, lo, hi):
    ui.range_bar_html(v, lo, hi, "#000")

def test_icons_not_wrong_for_common_words():
    assert ui.icon_for("Absolute Eosinophil Count") != "🧪"   # contains 'ast'? no. guard
    assert ui.icon_for("Fasting Blood Sugar") == "🍬"
    assert ui.icon_for("Salt") != "🧪"
    assert ui.icon_for("Last result") != "🧪"


# ---------- pdf ----------
def pdf_ok(r, concern="x"):
    b = pdf_export.build_pdf(r, concern); assert b[:4] == b"%PDF"; return b

def test_pdf_demo_samples():
    pdf_ok(demo_explanation("flagged")); pdf_ok(demo_explanation("normal"))

def test_pdf_hostile_text():
    evil = '<b>&</b> <unclosed "quotes" ≥ → 😀 இ \x00 {x}'
    f = F(name=evil, quote=evil); f.plain_meaning = evil; f.unit = evil
    pdf_ok(E([f], headline=evil, summary=evil, concern_answer=evil, limits_note=evil,
             seek_care_note=evil, completeness_note=evil), concern=evil)

def test_pdf_empty_and_long():
    pdf_ok(E([]))
    fs = [F(f"Test {i}", 10 + i, 12, 15, "normal") for i in range(60)]
    for f in fs: V.verify(E([f])); f.plain_meaning = "word " * 400
    b = pdf_ok(E(fs)); assert len(b) > 5000

def test_pdf_odd_numbers():
    pdf_ok(E([F(num=0, lo=0, hi=0), F(num=1e12, lo=None, hi=None), F(num=-5, lo=None, hi=3)]))


# ---------- ai.analyze with fake Gemini ----------
class FakeResp:
    def __init__(self, text=None, parsed=None): self.text, self.parsed = text, parsed

def install_fake(monkeypatch, behaviour):
    calls = []
    class Models:
        def generate_content(self, model, contents, config):
            calls.append(model); return behaviour(model, len(calls))
    class Client:
        def __init__(self, **kw): self.models = Models()
    import google.genai as g
    monkeypatch.setattr(g, "Client", Client)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return calls

GOOD = E([F()], tests_seen=1)

def test_no_key_sample_only():
    r, demo = ai.analyze("c", None, report_text=SAMPLE_REPORT); assert demo
    with pytest.raises(ai.AnalysisError): ai.analyze("c", None, report_text="my own")

def test_no_input(monkeypatch):
    install_fake(monkeypatch, lambda m, n: FakeResp(parsed=GOOD))
    with pytest.raises(ai.AnalysisError): ai.analyze("c", "k")
    with pytest.raises(ai.AnalysisError): ai.analyze("c", "k", file_bytes=b"", report_text="  ")

def test_bad_extension(monkeypatch):
    install_fake(monkeypatch, lambda m, n: FakeResp(parsed=GOOD))
    with pytest.raises(ai.AnalysisError): ai.analyze("c", "k", file_bytes=b"x", file_name="a.exe")
    with pytest.raises(ai.AnalysisError): ai.analyze("c", "k", file_bytes=b"x", file_name="noext")

def test_happy_path_json_text(monkeypatch):
    install_fake(monkeypatch, lambda m, n: FakeResp(text=GOOD.model_dump_json()))
    r, demo = ai.analyze("c", "k", report_text="Hb 13"); assert not demo and r.findings

def test_503_then_success_same_model(monkeypatch):
    def b(m, n):
        if n == 1: raise Exception("503 UNAVAILABLE high demand")
        return FakeResp(parsed=GOOD)
    calls = install_fake(monkeypatch, b); ai.analyze("c", "k", report_text="x"); assert len(calls) == 2

def test_404_moves_on(monkeypatch):
    def b(m, n):
        if n == 1: raise Exception("404 NOT_FOUND model gone")
        return FakeResp(parsed=GOOD)
    calls = install_fake(monkeypatch, b); ai.analyze("c", "k", report_text="x"); assert len(calls) == 2

def test_thinking_rejected_then_ok_and_still_retries_busy(monkeypatch):
    def b(m, n):
        if n == 1: raise Exception("400 thinking budget not supported")
        if n == 2: raise Exception("503 unavailable")
        return FakeResp(parsed=GOOD)
    calls = install_fake(monkeypatch, b)
    ai.analyze("c", "k", report_text="x"); assert calls[0] == calls[1] == calls[2], calls

def test_thinking_flag_not_leaked_to_next_model(monkeypatch):
    seen = []
    class Models:
        def generate_content(self, model, contents, config):
            seen.append((model, getattr(config, "thinking_config", None)))
            if len(seen) == 1: raise Exception("400 thinking not supported")
            if len(seen) == 2: raise Exception("404 not_found")
            return FakeResp(parsed=GOOD)
    class Client:
        def __init__(self, **kw): self.models = Models()
    import google.genai as g; monkeypatch.setattr(g, "Client", Client)
    ai.analyze("c", "k", report_text="x", model="m1")
    assert seen[-1][1] is not None, "next model should start with the fast setting again"

def test_error_message_prefers_informative_error(monkeypatch):
    def b(m, n):
        if n == 1: raise Exception("404 NOT_FOUND")
        raise Exception("503 UNAVAILABLE high demand")
    install_fake(monkeypatch, b)
    with pytest.raises(ai.AnalysisError) as e: ai.analyze("c", "k", report_text="x")
    assert "busy" in str(e.value).lower()

def test_key_never_in_message(monkeypatch):
    install_fake(monkeypatch, lambda m, n: (_ for _ in ()).throw(Exception("bad SECRETKEY123 here")))
    with pytest.raises(ai.AnalysisError) as e: ai.analyze("c", "SECRETKEY123", report_text="x")
    assert "SECRETKEY123" not in str(e.value)

def test_malformed_json_and_none_text(monkeypatch):
    install_fake(monkeypatch, lambda m, n: FakeResp(text="{not json"))
    with pytest.raises(ai.AnalysisError): ai.analyze("c", "k", report_text="x")
    install_fake(monkeypatch, lambda m, n: FakeResp(text=None))
    with pytest.raises(ai.AnalysisError): ai.analyze("c", "k", report_text="x")

def test_total_time_is_capped(monkeypatch):
    import time as T
    clock = [0.0]
    monkeypatch.setattr(T, "monotonic", lambda: clock[0])
    def b(m, n):
        clock[0] += 80; raise Exception("timed out")
    calls = install_fake(monkeypatch, b)
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    with pytest.raises(ai.AnalysisError): ai.analyze("c", "k", report_text="x")
    assert len(calls) <= 3, f"{len(calls)} attempts x 80s is too long for a user to wait"

def test_empty_findings_from_ai_is_not_silent(monkeypatch):
    install_fake(monkeypatch, lambda m, n: FakeResp(parsed=E([], tests_seen=0)))
    try:
        r, _ = ai.analyze("c", "k", report_text="x")
    except ai.AnalysisError:
        return
    assert r.completeness_note

def test_ai_marks_everything_normal_but_values_out(monkeypatch):
    r0 = E([F(num=9.0, status="normal")], tests_seen=1)
    install_fake(monkeypatch, lambda m, n: FakeResp(parsed=r0))
    r, _ = ai.analyze("c", "k", report_text="x"); assert r.findings[0].status != "normal"


# ---------- app flows (Streamlit AppTest) ----------
from streamlit.testing.v1 import AppTest
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

def fresh():
    os.environ.pop("GEMINI_API_KEY", None)
    return AppTest.from_file(APP, default_timeout=60).run()

def btn(at, label):
    return next(b for b in at.button if label in b.label)

def test_app_sample_flow_toggle_and_clear():
    at = fresh(); assert not at.exception
    btn(at, "flagged values").click().run()
    btn(at, "Explain my report").click().run(); assert not at.exception
    assert any("Your results" in m.value for m in at.markdown)
    at.toggle[0].set_value(True).run(); assert not at.exception
    btn(at, "Explain another").click().run(); assert not at.exception
    assert at.session_state["step"] == 1

def test_app_normal_sample_all_clear():
    at = fresh()
    btn(at, "all-normal").click().run(); btn(at, "Explain my report").click().run()
    assert not at.exception and at.session_state["step"] == 2

def test_app_clear_everything_clears_pasted_text():
    at = fresh()
    at.text_area(key="pasted").set_value("some pasted report").run()
    btn(at, "Clear everything").click().run()
    assert at.text_area(key="pasted").value == "", "Clear everything must also clear pasted text"

def test_app_paste_without_key_shows_friendly_error():
    at = fresh()
    at.text_area(key="pasted").set_value("random text").run()
    btn(at, "Explain my report").click().run()
    assert not at.exception and at.error

def test_app_all_clear_with_missing_results_shows_warning():
    at = fresh()
    r = E([F(num=13.0)], tests_seen=9); V.check_completeness(r)
    at.session_state["result"] = r; at.session_state["step"] = 2; at.session_state["is_demo"] = False
    at.run(); assert not at.exception
    assert at.warning, "all-clear screen must show the completeness warning"
    assert not any("All clear" in m.value for m in at.markdown), "must not say All clear when results may be missing"

def test_app_zero_findings_not_all_clear():
    at = fresh()
    r = E([]); V.check_completeness(r)
    at.session_state["result"] = r; at.session_state["step"] = 2; at.session_state["is_demo"] = False
    at.run(); assert not at.exception
    assert not any("All clear" in m.value for m in at.markdown)


# ---------- follow-up questions ----------
def test_followup_blocks_medicine_and_diagnosis_without_calling_ai(monkeypatch):
    calls = install_fake(monkeypatch, lambda m, n: FakeResp(text="x"))
    for q in ("What dose of iron should I take?", "Should I stop taking my tablets?", "Do I have diabetes?"):
        assert ai.answer_followup(q, GOOD, [], "k") == ai.BLOCKED_REPLY
    assert calls == []

def test_followup_limit_length_and_empty(monkeypatch):
    install_fake(monkeypatch, lambda m, n: FakeResp(text="ok"))
    with pytest.raises(ai.AnalysisError): ai.answer_followup("q", GOOD, [("a", "b")] * 5, "k")
    with pytest.raises(ai.AnalysisError): ai.answer_followup("x" * 500, GOOD, [], "k")
    with pytest.raises(ai.AnalysisError): ai.answer_followup("  ", GOOD, [], "k")
    with pytest.raises(ai.AnalysisError): ai.answer_followup("What is Hb?", GOOD, [], None)

def test_followup_answer_and_retry(monkeypatch):
    def b(m, n):
        if n == 1: raise Exception("503 unavailable")
        return FakeResp(text="Haemoglobin carries oxygen.")
    install_fake(monkeypatch, b)
    assert "oxygen" in ai.answer_followup("What is Hb?", GOOD, [], "k")

def test_followup_failure_is_friendly_and_hides_key(monkeypatch):
    install_fake(monkeypatch, lambda m, n: (_ for _ in ()).throw(Exception("boom KEY999")))
    with pytest.raises(ai.AnalysisError) as e: ai.answer_followup("What is Hb?", GOOD, [], "KEY999")
    assert "KEY999" not in str(e.value)

def test_followup_context_only_uses_report():
    c = ai.followup_context(GOOD); assert "Haemoglobin" in c and "printed range" in c

def test_app_followup_ui_shows_when_results():
    at = fresh()
    btn(at, "flagged values").click().run(); btn(at, "Explain my report").click().run()
    assert not at.exception and at.session_state["followups"] == []
    assert any("follow-up" in m.value.lower() for m in at.markdown)
    # blocked question gets the safe reply, counts as one of five
    at.text_input[0].set_value("What dose should I take?")
    at.button[[b.label for b in at.button].index("Ask")].click().run()
    assert not at.exception and len(at.session_state["followups"]) == 1
    btn(at, "Explain another").click().run(); assert at.session_state["followups"] == []
