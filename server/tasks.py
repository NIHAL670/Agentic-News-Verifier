from typing import Dict, Any, List

MIN_VALID_SCORE = 0.01
MAX_VALID_SCORE = 0.99


def _safe_score(value: float) -> float:
    """Clamp score to strictly (0, 1) — never exactly 0.0 or 1.0."""
    if value is None:
        return MIN_VALID_SCORE
    value = float(value)
    if value <= 0.0:
        return MIN_VALID_SCORE
    if value >= 1.0:
        return MAX_VALID_SCORE
    return round(value, 4)


def fake_news_grader(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """
    Grade the agent's output against expected output.
    Returns a score strictly in (0, 1).

    Scoring breakdown:
    - Correct label match: +0.6
    - Required keyword found in evidence: +0.38
    - Base score: 0.01 (never returns exactly 0)
    """
    if not isinstance(output, dict):
        return MIN_VALID_SCORE

    score = 0.0

    # Check label match
    out_label = str(output.get("label", "")).strip().lower()
    exp_label = str(expected.get("label", "")).strip().lower()
    if out_label == exp_label:
        score += 0.6

    # Check for required keyword in evidence
    expected_keyword = str(expected.get("required_keyword", "")).strip().lower()
    if expected_keyword:
        out_evidence = str(output.get("evidence", "")).strip().lower()
        if expected_keyword in out_evidence:
            score += 0.38

    return max(0.01, min(0.99, score if score > 0 else 0.01))


tasks: List[Dict[str, Any]] = [
    {
        "id": "task-1",
        "name": "Verify headline 1",
        "difficulty": "easy",
        "max_steps": 5,
        "input": {
            "headline": "NASA confirms Moon is Swiss Cheese."
        },
        "expected_output": {
            "label": "false",
            "required_keyword": "science",
        },
        "grader": fake_news_grader,
    },
    {
        "id": "task-2",
        "name": "Verify headline 2",
        "difficulty": "medium",
        "max_steps": 5,
        "input": {
            "headline": "New policy: 1000 units for all tomorrow."
        },
        "expected_output": {
            "label": "true",
            "required_keyword": "gazette",
        },
        "grader": fake_news_grader,
    },
    {
        "id": "task-3",
        "name": "Verify headline 3",
        "difficulty": "hard",
        "max_steps": 5,
        "input": {
            "headline": "Coffee leads to 20% IQ increase."
        },
        "expected_output": {
            "label": "false",
            "required_keyword": "study",
        },
        "grader": fake_news_grader,
    },
]
