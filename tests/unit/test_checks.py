"""Deterministic check primitives that read the request sources."""

import json

import pytest

from kairyu.orchestration.checks import CheckContext, CheckParameterError, run_check

SOURCES = "Revenue in FY2025 was 1,234 million yen. Tool result: deploy finished at 12:05."


@pytest.mark.parametrize(
    ("primitive", "params", "text", "passed"),
    [
        ("quotes_in_sources", {}, "The report says 「Revenue in FY2025 was 1,234」.", True),
        ("quotes_in_sources", {}, "The report says 「Revenue in FY2025 doubled」.", False),
        # Typography differs, wording is verbatim.
        ("quotes_in_sources", {}, "It says \u201cRevenue in FY2025 was 1,234\u201d.", True),
        # String literals inside code are not quotations.
        ("quotes_in_sources", {}, 'Use `print("hello wide world")` here.', True),
        # A JSON answer's keys and strings are not quotations.
        ("quotes_in_sources", {}, '{"city": "Tokyo", "population_estimate": 14000000}', True),
        ("numbers_in_sources", {}, "Revenue was 1234 million yen in 2025.", False),
        ("numbers_in_sources", {}, "Revenue was 1,234 million yen.", True),
        ("numbers_in_sources", {"ignore_below": 10}, "It took 3 steps.", True),
    ],
)
def test_source_backed_primitives(primitive, params, text, passed):
    ctx = CheckContext(text=text, sources=SOURCES, outputs={})
    assert run_check(primitive, params, ctx).passed is passed


def test_items_in_sources_reads_another_roles_json():
    claims = json.dumps(
        {
            "claims": [
                {"id": "c1", "action": True, "evidence": "deploy finished at 12:05"},
                {"id": "c2", "action": True, "evidence": "tests passed"},
                {"id": "c3", "action": False, "evidence": "anything"},
            ]
        }
    )
    ctx = CheckContext(text="", sources=SOURCES, outputs={"state": claims})
    outcome = run_check(
        "items_in_sources",
        {"role": "state", "path": "claims", "where": {"action": True}, "key": "evidence"},
        ctx,
    )
    assert not outcome.passed and "c2" in outcome.detail and "c1" not in outcome.detail


def test_unknown_primitive_is_rejected():
    with pytest.raises(CheckParameterError):
        run_check("guess", {}, CheckContext(text="", sources="", outputs={}))


def test_execution_evidence_counts_only_tool_results():
    conversation = (
        "--- CONVERSATION CONTEXT JSON ---\n"
        + json.dumps(
            [
                {"role": "assistant", "content": "I ran the test suite and all tests passed."},
                {"role": "tool", "content": "pytest: 12 passed in 0.4s"},
            ]
        )
        + "\n--- END CONVERSATION CONTEXT JSON ---"
    )
    params = {
        "role": "state",
        "path": "claims",
        "key": "evidence",
        "message_roles": ["tool"],
    }

    def outcome(evidence: str) -> bool:
        claims = json.dumps({"claims": [{"id": "c1", "evidence": evidence}]})
        ctx = CheckContext(text="", sources=conversation, outputs={"state": claims})
        return run_check("items_in_sources", params, ctx).passed

    assert outcome("pytest: 12 passed") is True
    # An earlier assistant statement is not proof that anything ran.
    assert outcome("I ran the test suite and all tests passed.") is False
