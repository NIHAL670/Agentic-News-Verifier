from typing import Dict, Any, List

def fake_news_grader(output: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """
    Expected output example:
    {
        "label": "true" or "false",
        "evidence": "...",
    }
    """
    if not isinstance(output, dict):
        return False

    out_label = str(output.get("label", "")).strip().lower()
    exp_label = str(expected.get("label", "")).strip().lower()

    if out_label != exp_label:
        return False

    # Optional evidence check if provided
    expected_keyword = str(expected.get("required_keyword", "")).strip().lower()
    if expected_keyword:
        out_evidence = str(output.get("evidence", "")).strip().lower()
        if expected_keyword not in out_evidence:
            return False

    return True


tasks: List[Dict[str, Any]] = [
    {
        "id": "task-1",
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
