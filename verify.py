"""Safeguard that runs in plain code AFTER the AI answers.

The AI reads the report and writes the explanation, but it does not get the final say on two things:
1. Status (normal / borderline / attention): recomputed from the value and the report's own printed range.
2. Value: the number the AI reports must appear in the line it quotes from the report. If not, the
   result is marked 'check this against your report' so the user can catch a misread.
"""
import re

_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _numbers(text: str) -> list[float]:
    out = []
    for tok in _NUM.findall(text or ""):
        tok = tok.replace(",", "").rstrip(".")
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _outside(v, low, high):
    """Returns (is_outside, relative_distance_outside_the_range)."""
    if low is not None and v < low:
        return True, (low - v) / abs(low) if low else 1.0
    if high is not None and v > high:
        return True, (v - high) / abs(high) if high else 1.0
    return False, 0.0


def verify(result) -> int:
    """Corrects statuses in place, adds check_note where needed. Returns how many findings were changed."""
    changed = 0
    for f in result.findings:
        f.check_note = None  # never trust a note written by the AI
        v = f.numeric_value
        if v is None:
            continue

        # 1. the value must appear in the quoted report line
        if f.source_quote:
            nums = _numbers(f.source_quote)
            if nums and not any(abs(n - v) <= max(1e-9, abs(v) * 1e-6) for n in nums):
                f.check_note = ("This value may have been misread. Please compare it with your original report "
                                "before relying on this explanation.")

        # 2. status must agree with the printed range
        if f.ref_low is None and f.ref_high is None:
            continue
        lo, hi = f.ref_low, f.ref_high
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo  # the AI swapped the ends of the range
        outside, dist = _outside(v, lo, hi)
        new = f.status
        if outside and f.status == "normal":
            new = "attention" if dist > 0.25 else "borderline"
        elif not outside and f.status == "attention":
            new = "borderline"
        if new != f.status:
            f.status = new
            changed += 1
            extra = "Status re-checked against the range printed in your report."
            f.check_note = (f.check_note + " " + extra) if f.check_note else extra
    return changed


def check_completeness(result, rechecked: bool = False) -> None:
    """If the AI counted more test rows than it listed, tell the user so they never assume the list is complete."""
    result.completeness_note = None
    seen, listed = getattr(result, "tests_seen", 0) or 0, len(result.findings)
    if listed == 0:
        result.completeness_note = ("No individual results could be listed from this document. Please try again, "
                                    "or use a clearer file, and check the original report yourself.")
    elif seen and listed < seen:
        extra = (" A second automatic check did not find the missing ones, so the count may simply be off by a little."
                 if rechecked else "")
        result.completeness_note = (f"This explanation lists {listed} of the roughly {seen} results the AI found in your report."
                                    f"{extra} Some results may be missing, so please check the original report for the rest. "
                                    f"Try again, or upload the pages separately, if you need every result explained.")
