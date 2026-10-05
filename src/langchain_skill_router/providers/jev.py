"""Adapter for Jev (TypeSafe): turns our questions into its choices and nouls, all in one call."""

import logging
from collections.abc import Callable, Mapping
from typing import cast

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    JSONContent,
    Noul,
    NoulAnswer,
    RetryPolicy,
    TypeSafeAuthenticationError,
    TypeSafePermissionDeniedError,
)

from langchain_skill_router.core.judge import Answer, JudgeMisconfigured, JudgeUnavailable, Limits, Pick, YesNo

logger = logging.getLogger(__name__)

# Credentials that the provider rejects: retrying cannot help, so these surface instead of falling back.
# A bad request, including one that is too large, is deliberately not here: the router answers that by
# splitting the catalog and asking again.
PERMANENT = (TypeSafeAuthenticationError, TypeSafePermissionDeniedError)


class JevJudge:
    # 32k tokens for the state plus the longest question, up to 255 options (docs.typesafe.ai/models).
    # Tokens per character: Russian descriptions measure about 0.68, so 0.8 leaves a margin.
    limits = Limits(max_tokens=32_000, max_options=255, tokens_per_char=0.8)

    def __init__(
        self,
        client: AsyncTypeSafeClient | None = None,
        on_usage: Callable[[int], None] | None = None,
        timeout: float | None = None,
    ) -> None:
        """`timeout` is the limit on each call, in seconds (see `Judge`): it goes into every request, over the
        client's own setting. Left out, a call keeps the client's timeout, 10 s by default in the SDK.
        Exceptions from `on_usage` are logged once per judge without discarding the answer; cancellation propagates."""
        self.timeout = timeout
        # No retries: a decision has two seconds, and a late answer is useless because the fallback has run.
        self.client = client or AsyncTypeSafeClient(retry=RetryPolicy(max_retries=0), timeout=timeout)
        self.on_usage = on_usage  # tokens per call, so cost can be attributed to decisions under concurrency
        self._usage_warning_logged = False

    async def ask(self, state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        asked = {key: _to_jev(q) for key, q in questions.items()}
        try:
            # The port requires the state to be JSON-serialisable, which is what JSONContent means.
            response = await self.client.system_one(cast(JSONContent, dict(state)), asked, timeout=self.timeout)
        except PERMANENT as err:
            raise JudgeMisconfigured(f"Jev rejected the credentials: {err}") from err
        except Exception as err:
            raise JudgeUnavailable(f"Jev did not answer: {err}") from err
        if self.on_usage:
            tokens = response.usage.input_tokens or 0
            try:
                self.on_usage(tokens)
            except Exception as err:
                # User metrics must not discard an answer. Their error text may contain private data.
                # One decision can call the judge repeatedly; a broken callback must not flood the log.
                if not self._usage_warning_logged:
                    self._usage_warning_logged = True
                    logger.warning("Jev on_usage callback failed (%s); keeping the answer", type(err).__name__)
        return {key: _from_jev(answer) for key, answer in response.answers.items()}


def _to_jev(q: Pick | YesNo) -> Choice | Noul:
    if isinstance(q, Pick):
        return Choice(instructions=q.instructions, criteria=dict(q.options))
    return Noul(instructions=q.instructions)


def _from_jev(answer: object) -> Answer:
    if isinstance(answer, ChoiceAnswer):
        return Answer(dict(answer.probabilities))
    if isinstance(answer, NoulAnswer):
        return Answer({"yes": answer.noul})
    raise TypeError(f"unexpected answer from Jev: {type(answer).__name__}")
