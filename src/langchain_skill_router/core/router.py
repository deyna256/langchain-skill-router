"""The skill router: every decision rule lives here, and the provider sits behind the `Judge` port.

A turn goes through:

1. Ranking, with "is a skill needed at all" asked alongside it: pick from the catalog by name and
   description, keeping the best `max_candidates`. If the catalog does not fit the provider's limits
   (`judge.limits`), it is split into parts that do, keeping groups together; the parts are asked in
   parallel and their winners merged by a further pick, repeating until everything fits one call.
   A low "need" ends the decision here, with nothing loaded.
2. Verification, one call over the heads of the candidates' texts: which of them is the right one, and
   for each on its own, does it do what the request asks. The pick settles which skill, and how sure it is
   settles whether it is loaded or only offered; `fits` admits a candidate at all. With ranking confident in
   its first candidate (`skip_verify_at`), this step is skipped.
3. Thresholds from `Settings`, applied by `decide_from_trace`.

A turn is decided on its own. Nothing is carried over from the turn before it; why not is on `Turn`.

The questions are written in English, which is what the judges measured so far answer best in; the
user's own request is passed through unchanged, in whatever language it arrives. A product can replace
the one question that depends on its domain through `Settings.need_question`.
"""

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace

from langchain_skill_router.core.judge import Answer, Judge, JudgeMisconfigured, JudgeUnavailable, Limits, Pick, YesNo
from langchain_skill_router.core.types import DEFAULTS, Decision, Settings, Skill, Trace, Turn

# Every question the judge is asked lives in `Settings`, so a product can write them for its own domain.
# A candidate's text goes in its own question rather than in the shared state. That way its score does
# not depend on its neighbours in the same call (measured: a shift of up to 0.46 with neighbours in the
# state, no more than 0.05 this way), and a call costs the state plus one longest question rather than
# every candidate at once.
SPARE_CHARS = 1200  # room for the question texts themselves when checking whether settings fit

Ranking = list[tuple[str, float]]


def decide_from_trace(s: Settings, trace: Trace) -> Decision:
    """Thresholds to a decision. Kept apart from the calls, so thresholds can be refitted on recorded
    traces without asking the provider again."""
    if trace.need is None or trace.need < s.need_at:
        return Decision(trace=trace)
    top = next(iter(trace.candidates), None)
    if s.skip_verify_at is not None and top and top[1] >= s.skip_verify_at:
        if s.max_load == 0:
            return Decision(suggest=(top[0],)[: s.max_suggest], trace=trace)
        return Decision(load=(top[0],), trace=trace)  # ranking is sure; skip verification
    # Verification's pick orders the candidates; a trace recorded without one falls back to the ranking.
    # `fits` does not order them: independent yes/no answers tie on lookalikes.
    ranked = [n for n, _ in trace.candidates]
    order = sorted(trace.fits, key=lambda n: (-trace.picked.get(n, 0.0), ranked.index(n) if n in ranked else len(ranked)))
    plausible = [n for n in order if trace.fits[n] >= s.suggest_at]
    # How sure verification is of a candidate decides loading; `fits` only admits it. With a single candidate
    # there is no pick, and its fit stands in.
    sure = trace.picked if trace.picked else trace.fits
    load = tuple(n for n in plausible[: s.max_load] if sure.get(n, 0.0) >= s.load_at)
    suggest = tuple(n for n in plausible if n not in load)[: s.max_suggest]
    return Decision(load=load, suggest=suggest, trace=trace)


def call_size(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> int:
    """The size of a call in characters, erring high: measured both as JSON and as text, the larger wins."""
    state_chars = max(len(json.dumps(state, ensure_ascii=False)), len(str(state)))
    return state_chars + sum(
        len(q.instructions) + sum(len(k) + len(v) + 6 for k, v in getattr(q, "options", {}).items()) for q in questions.values()
    )


class SkillRouter:
    def __init__(self, catalog: Sequence[Skill], judge: Judge, settings: Settings = DEFAULTS) -> None:
        names = [s.name for s in catalog]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate skill names in the catalog: {sorted({n for n in names if names.count(n) > 1})}")
        if empty := [s.name for s in catalog if not s.description.strip()]:
            raise ValueError(f"skills without a description: {empty}")
        self.skills = {s.name: s for s in catalog}
        self.judge = judge
        self.settings = settings
        # The protocol declares `limits`, but nothing enforces a Protocol at run time, so an adapter
        # that forgets it is caught here rather than deep inside catalog splitting.
        limits = getattr(judge, "limits", None)
        if not isinstance(limits, Limits):
            raise JudgeMisconfigured(
                f"{type(judge).__name__} does not declare limits; add `limits = Limits(...)`, or `Limits()` if the provider has none"
            )
        self.limits: Limits = limits
        timeout = getattr(judge, "timeout", None)  # optional in the protocol: absent means no per-call limit
        if timeout is not None and not timeout > 0:
            raise JudgeMisconfigured(f"{type(judge).__name__}.timeout is {timeout}; expected a positive number of seconds or None")
        self.call_timeout: float | None = timeout
        self.budget = self.limits.max_tokens * settings.budget_share / self.limits.tokens_per_char  # characters per call
        # A provider's limit covers the state plus the longest question. In verification that is the pick, which
        # carries every candidate's description and head as its options; a `fits` question carries one head.
        option = max((len(s.name) + len(s.description) for s in catalog), default=0) + settings.head_chars
        longest_question = max(len(settings.pick_question) + settings.max_candidates * option, settings.head_chars)
        check = settings.request_chars + settings.context_chars + longest_question + SPARE_CHARS
        if check > self.budget:
            raise JudgeMisconfigured(
                f"settings do not fit the provider's limit: verification needs about {check} characters, "
                f"the limit is about {int(self.budget)}; reduce max_candidates, head_chars, context_chars or request_chars"
            )

    async def decide(self, turn: Turn) -> Decision:
        """Decide one turn. Any failure means changing nothing, or suggesting the ranked candidates if it was
        verification that failed."""
        if not self.skills:
            return Decision(trace=Trace(stage="empty"))
        started = time.monotonic()
        # `_decide` appends to this as soon as ranking finishes. The timeout below fires outside `_decide`,
        # so this is how the handler learns what had already been ranked when the clock ran out.
        ranked_so_far: list[str] = []
        try:
            async with asyncio.timeout(self.settings.timeout):
                return await self._decide(turn, started, ranked_so_far)
        except JudgeMisconfigured:
            raise  # retrying will not help, and a hidden broken setup is worse than a loud one
        except Exception as err:
            failure = "timeout" if isinstance(err, TimeoutError) else f"{type(err).__name__}: {err}"
            trace = Trace(failure=failure, seconds=time.monotonic() - started)
            return Decision(suggest=tuple(ranked_so_far[: self.settings.max_suggest]), trace=trace)

    async def _decide(self, turn: Turn, started: float, ranked_so_far: list[str]) -> Decision:
        s = self.settings
        base = {"request": turn.request[: s.request_chars], "context": turn.context[-s.context_chars :]}

        def elapsed() -> float:
            return time.monotonic() - started

        # "Is a skill needed at all" goes out alongside the ranking rather than before it. Asked first it
        # would end about one turn in fifty on its own and cost every other turn a round trip; asked in
        # parallel it costs neither.
        gate, ranked = await asyncio.gather(
            self._ask(base, {"need": YesNo(s.need_question)}),
            self._rank(base, list(self.skills), s.max_candidates),
        )
        ranking, parts = ranked
        trace = Trace(need=gate["need"].yes, parts=parts, stage="gate")
        if trace.need is not None and trace.need < s.need_at:
            return decide_from_trace(s, replace(trace, seconds=elapsed()))

        candidates = ranking[: s.max_candidates]
        ranked_so_far += [n for n, _ in candidates]
        trace = replace(trace, candidates=tuple(candidates))

        # Verify candidates against their texts, unless the product allows skipping a confident ranking.
        if s.skip_verify_at is not None and candidates and candidates[0][1] >= s.skip_verify_at:
            return decide_from_trace(s, replace(trace, stage="skip", seconds=elapsed()))
        heads = await self._read_heads(ranked_so_far)
        questions: dict[str, Pick | YesNo] = {
            f"fits:{n}": YesNo(s.fits_question.format(name=n, text=head)) for n, head in heads.items()
        }
        if len(heads) > 1:
            options = {n: f"{self.skills[n].description}\n\n{head}" for n, head in heads.items()}
            questions["pick"] = Pick(s.pick_question, options)
        answers = await self._ask(base, questions) if questions else {}
        fits = {n: answers[f"fits:{n}"].yes for n in heads}
        picked = {n: p for n, p in answers["pick"].probabilities.items() if n in heads} if "pick" in answers else {}
        return decide_from_trace(s, replace(trace, fits=fits, picked=picked, stage="verify", seconds=elapsed()))

    async def search(self, query: str, limit: int = 5) -> list[Skill]:
        """Back the `find_skill` tool: the best skills for the model's own query, or nothing on failure."""
        if not self.skills:
            return []
        try:
            async with asyncio.timeout(self.settings.timeout):
                ranking, _ = await self._rank({"request": query[: self.settings.request_chars]}, list(self.skills), limit)
        except JudgeMisconfigured:
            raise
        except Exception:
            return []
        return [self.skills[n] for n, _ in ranking][:limit]

    # --- ranking that fits the provider's limits ----------------------------------------------------

    async def _rank(self, state: Mapping[str, object], names: list[str], k: int) -> tuple[Ranking, int]:
        """The best skills among `names`, best first. Too big for one call means parts, then a merge."""
        parts = self._split_catalog(state, names)
        answers = await asyncio.gather(*(self._ask_part(state, p) for p in parts))
        rankings = [r for got, _ in answers for r in got]
        if not rankings:
            raise next(err for _, err in answers if err is not None)  # the provider's own error belongs in the trace
        if len(rankings) == 1:
            return rankings[0], len(parts)
        # take the best k from each part, but fewer than the part holds, so the merge round is smaller
        best = [n for ranking in rankings for n, _ in ranking[: max(1, min(k, len(ranking) - 1))]]
        if len(best) >= len(names):
            raise JudgeMisconfigured(
                "the catalog cannot be split to fit the provider's limits: a single skill's name and "
                "description are larger than one call allows"
            )
        merged, _ = await self._rank(state, best, k)
        return merged, len(parts)

    async def _ask_part(self, state: Mapping[str, object], names: list[str]) -> tuple[list[Ranking], Exception | None]:
        """Pick within one part. A refusal is retried as two halves, which covers an underestimated size.
        Returns the rankings obtained, and the original error when none were."""
        try:
            return [await self._pick(state, names)], None
        except JudgeMisconfigured:
            raise
        except Exception as err:
            if len(names) < 2:
                return [], err
            halves = [names[: len(names) // 2], names[len(names) // 2 :]]
            results = await asyncio.gather(*(self._pick(state, h) for h in halves), return_exceptions=True)
            ok = [r for r in results if not isinstance(r, BaseException)]
            return ok, None if ok else err

    async def _ask(self, state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        """One call to the judge, cut at the judge's own `timeout`. A cut call is unavailable, like any other
        provider failure; the decision's timeout, when it fires, passes through untouched."""
        try:
            async with asyncio.timeout(self.call_timeout):
                return await self.judge.ask(state, questions)
        except TimeoutError as err:
            raise JudgeUnavailable(f"the judge did not answer within {self.call_timeout} s") from err

    async def _pick(self, state: Mapping[str, object], names: list[str]) -> Ranking:
        options = {n: self.skills[n].description for n in names}
        picked = await self._ask(state, {"skill": Pick(self.settings.rank_question, options)})
        ranked = sorted(picked["skill"].probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return [(n, p) for n, p in ranked if n in options]

    def _split_catalog(self, state: Mapping[str, object], names: list[str]) -> list[list[str]]:
        """Parts that each fit the limit, keeping skills of one group together where possible."""
        fixed = call_size(state, {"skill": Pick(self.settings.rank_question, {})})
        room = self.budget - fixed
        cost = {n: len(n) + len(self.skills[n].description) + 6 for n in names}
        if sum(cost.values()) <= room and len(names) <= self.limits.max_options:
            return [names]
        groups: dict[str, list[str]] = {}
        for n in names:
            groups.setdefault(self.skills[n].group or f"_{n}", []).append(n)
        parts: list[list[str]] = [[]]
        for members in groups.values():
            for part in self._split_group(members, cost, room):
                if parts[-1] and not self._fits_one_call(parts[-1] + part, cost, room):
                    parts.append([])
                parts[-1] += part
        return [p for p in parts if p]

    def _fits_one_call(self, names: list[str], cost: Mapping[str, int], room: float) -> bool:
        return sum(cost[n] for n in names) <= room and len(names) <= self.limits.max_options

    def _split_group(self, names: list[str], cost: Mapping[str, int], room: float) -> list[list[str]]:
        """Split `names` into consecutive chunks that each fit `room`."""
        parts: list[list[str]] = [[]]
        for n in names:
            if parts[-1] and not self._fits_one_call(parts[-1] + [n], cost, room):
                parts.append([])
            parts[-1].append(n)
        return parts

    async def _read_heads(self, names: list[str]) -> dict[str, str]:
        """The head of each candidate's text. A skill that cannot be read drops out; the rest are verified."""
        texts = await asyncio.gather(*(self.skills[n].read() for n in names), return_exceptions=True)
        return {n: t[: self.settings.head_chars] for n, t in zip(names, texts, strict=True) if isinstance(t, str)}
