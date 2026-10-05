"""The Jev adapter: how it translates questions and how it classifies provider failures."""

import asyncio
from typing import Any

import httpx2
import pytest
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeInternalServerError,
    TypeSafePermissionDeniedError,
)

from langchain_skill_router import JudgeMisconfigured, JudgeUnavailable, Limits, Pick, YesNo
from langchain_skill_router.providers.jev import JevJudge


def _choice(criteria: dict[str, str]) -> ChoiceAnswer:
    """An even distribution over the options, shaped the way the SDK returns one."""
    return ChoiceAnswer(choice=next(iter(criteria)), confidence=0.5, probabilities=dict.fromkeys(criteria, 0.5))


class Client:
    """Stands in for the SDK client: records the call, then answers or raises."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.raises = raises
        self.asked: Any = None

    async def system_one(self, state: Any, questions: Any, **kwargs: Any) -> Any:
        self.asked = (state, questions)
        self.kwargs = kwargs
        if self.raises:
            raise self.raises
        answers = {key: _choice(q.criteria) if isinstance(q, Choice) else NoulAnswer(noul=0.75) for key, q in questions.items()}
        return type("Response", (), {"answers": answers, "usage": type("Usage", (), {"input_tokens": 42})()})()


def api_error(kind: type[Exception]) -> Exception:
    return kind(status=401, body=None, headers=httpx2.Headers())


def test_limits_match_what_the_provider_documents():
    assert JevJudge(Client()).limits == Limits(max_tokens=32_000, max_options=255, tokens_per_char=0.8)


async def test_questions_are_translated_and_answers_come_back_in_our_shape():
    client = Client()
    judge = JevJudge(client)
    answers = await judge.ask(
        {"request": "spending"},
        {"skill": Pick("Which one?", {"a": "first", "b": "second"}), "need": YesNo("Is a skill needed?")},
    )
    _, asked = client.asked

    assert isinstance(asked["skill"], Choice) and asked["skill"].criteria == {"a": "first", "b": "second"}
    assert isinstance(asked["need"], Noul)
    assert answers["skill"].probabilities == {"a": 0.5, "b": 0.5}
    assert answers["need"].yes == 0.75


async def test_token_usage_is_reported_for_every_call():
    spent: list[int] = []

    await JevJudge(Client(), on_usage=spent.append).ask({}, {"need": YesNo("Is a skill needed?")})

    assert spent == [42]


@pytest.mark.parametrize("kind", [KeyError, RuntimeError])
async def test_usage_callback_failure_preserves_answers_and_logs_without_payload(kind, caplog):
    def on_usage(tokens):
        assert tokens == 42
        raise kind("private metrics label")

    questions = {"skill": Pick("Which one?", {"a": "first", "b": "second"}), "need": YesNo("Is a skill needed?")}
    expected = await JevJudge(Client()).ask({}, questions)
    actual = await JevJudge(Client(), on_usage=on_usage).ask({}, questions)

    assert actual == expected
    assert len(caplog.records) == 1
    assert caplog.records[0].name == "langchain_skill_router.providers.jev"
    assert caplog.records[0].levelname == "WARNING"
    assert "on_usage" in caplog.text
    assert kind.__name__ in caplog.text
    assert "private metrics label" not in caplog.text
    assert caplog.records[0].exc_info is None


async def test_usage_callback_cancellation_propagates():
    def on_usage(tokens):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await JevJudge(Client(), on_usage=on_usage).ask({}, {"need": YesNo("Is a skill needed?")})


async def test_usage_callback_warns_once_per_judge_but_keeps_reporting_usage(caplog):
    spent = []

    def on_usage(tokens):
        spent.append(tokens)
        raise RuntimeError("private metrics label")

    judge = JevJudge(Client(), on_usage=on_usage)
    questions = {"need": YesNo("Is a skill needed?")}
    answers = await asyncio.gather(*(judge.ask({}, questions) for _ in range(3)))

    assert all(answer["need"].yes == 0.75 for answer in answers)
    assert spent == [42, 42, 42]
    assert len(caplog.records) == 1

    await JevJudge(Client(), on_usage=on_usage).ask({}, questions)
    assert len(caplog.records) == 2


@pytest.mark.parametrize("kind", [TypeSafeAuthenticationError, TypeSafePermissionDeniedError])
async def test_rejected_credentials_are_misconfiguration(kind):
    judge = JevJudge(Client(raises=api_error(kind)))
    with pytest.raises(JudgeMisconfigured, match="credentials"):
        await judge.ask({}, {"need": YesNo("Is a skill needed?")})


@pytest.mark.parametrize("kind", [TypeSafeInternalServerError, TypeSafeBadRequestError])
async def test_everything_else_is_unavailable_so_the_router_can_retry_or_fall_back(kind):
    judge = JevJudge(Client(raises=api_error(kind)))
    with pytest.raises(JudgeUnavailable):
        await judge.ask({}, {"need": YesNo("Is a skill needed?")})


async def test_an_unexpected_answer_type_is_reported():
    class Odd(Client):
        async def system_one(self, state, questions, **kwargs):
            return type("Response", (), {"answers": {"need": object()}, "usage": type("U", (), {"input_tokens": 0})()})()

    with pytest.raises(TypeError, match="unexpected answer"):
        await JevJudge(Odd()).ask({}, {"need": YesNo("Is a skill needed?")})


async def test_the_judges_timeout_goes_into_every_request_over_the_clients_own():
    client = Client()

    await JevJudge(client, timeout=50.0).ask({}, {"need": YesNo("Is a skill needed?")})
    assert client.kwargs["timeout"] == 50.0

    await JevJudge(client).ask({}, {"need": YesNo("Is a skill needed?")})
    assert client.kwargs["timeout"] is None  # the client keeps its own
