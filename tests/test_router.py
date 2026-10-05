"""What the router decides on a turn, given the answers the judge is scripted to give."""

from collections.abc import Mapping
from dataclasses import replace

import pytest
from conftest import skill

from langchain_skill_router import (
    Answer,
    JudgeMisconfigured,
    Pick,
    Settings,
    Skill,
    SkillRouter,
    Trace,
    Turn,
    YesNo,
    decide_from_trace,
)
from langchain_skill_router.testing import ScriptedJudge, yes

CATALOG = [skill(n) for n in ("visa-statement", "spending-by-category", "subscriptions", "card-limits", "dispute")]


async def test_empty_catalog_returns_empty_decisions_without_asking_the_judge():
    judge = ScriptedJudge(lambda state, questions: {})
    router = SkillRouter([], judge)

    for turn in (Turn("hello"), Turn("and now?", context="previous conversation")):
        decision = await router.decide(turn)
        assert (decision.load, decision.suggest) == ((), ())
        assert decision.trace.failure is None
        assert decision.trace.stage == "empty"
        assert decision.trace.candidates == ()
    assert judge.calls == []


def scripted(
    pick: Mapping[str, float],
    need: float = 0.9,
    fits: Mapping[str, float] | None = None,
    choose: Mapping[str, float] | None = None,
) -> ScriptedJudge:
    """A scripted judge: ranking probabilities, the need answer, a fit per candidate and, when given,
    verification's pick among the candidates (otherwise a sure pick of the ranking's best candidate)."""

    def answer(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        out: dict[str, Answer] = {}
        for key, q in questions.items():
            if isinstance(q, Pick):
                if key == "pick":
                    best = max(q.options, key=lambda o: pick.get(o, 0.0))
                    probabilities = choose if choose is not None else {best: 0.95}
                else:
                    probabilities = pick
                out[key] = Answer({o: probabilities.get(o, 0.0) for o in q.options})
            elif key == "need":
                out[key] = yes(need)
            elif key.startswith("fits:"):
                out[key] = yes((fits or {}).get(key.removeprefix("fits:"), 0.0))
        return out

    return ScriptedJudge(answer)


async def test_no_skill_needed_loads_nothing():
    judge = scripted({"visa-statement": 0.5}, need=0.1, fits={"visa-statement": 0.95})

    d = await SkillRouter(CATALOG, judge).decide(Turn("hello"))

    assert (d.load, d.suggest, d.trace.stage) == ((), (), "gate")


async def test_the_need_question_goes_out_alongside_the_ranking_not_before_it():
    judge = scripted({"visa-statement": 0.5}, need=0.1)

    await SkillRouter(CATALOG, judge).decide(Turn("thanks"))
    asked = [set(questions) for _, questions in judge.calls]

    assert {"need"} in asked  # the cheap question
    assert any("skill" in keys for keys in asked)  # and the catalog, in the same round
    assert len(judge.calls) == 2  # verification never runs


async def test_confident_fit_is_loaded_doubtful_is_suggested():
    judge = scripted({"visa-statement": 0.6, "spending-by-category": 0.3}, fits={"visa-statement": 0.92, "spending-by-category": 0.55})

    d = await SkillRouter(CATALOG, judge).decide(Turn("I need a statement for the embassy"))

    assert d.load == ("visa-statement",)
    assert d.suggest == ("spending-by-category",)


async def test_verification_picks_which_skill_and_fits_only_decides_whether():
    # Lookalikes both "fit"; the pick over their texts side by side is what tells them apart.
    fits = {"visa-statement": 0.85, "spending-by-category": 0.9}
    choose = {"visa-statement": 0.92, "spending-by-category": 0.08}
    judge = scripted({"visa-statement": 0.6, "spending-by-category": 0.3}, fits=fits, choose=choose)

    d = await SkillRouter(CATALOG, judge).decide(Turn("I need a statement for the embassy"))

    assert d.load == ("visa-statement",)
    assert d.suggest == ("spending-by-category",)
    assert d.trace.picked["visa-statement"] == 0.92


async def test_verification_reads_the_candidates_texts_in_its_pick():
    judge = scripted({"visa-statement": 0.6, "spending-by-category": 0.3}, fits={"visa-statement": 0.9})

    await SkillRouter(CATALOG, judge, Settings(max_candidates=2)).decide(Turn("statement"))
    _, _, verify = judge.calls

    assert set(verify[1]["pick"].options) == {"visa-statement", "spending-by-category"}
    assert "Instructions for visa-statement." in verify[1]["pick"].options["visa-statement"]


async def test_an_unsure_pick_is_offered_rather_than_loaded():
    # Both fit well; verification cannot tell them apart, so the model is given the choice.
    fits = {"visa-statement": 0.95, "spending-by-category": 0.3}
    choose = {"visa-statement": 0.6, "spending-by-category": 0.4}
    judge = scripted({"visa-statement": 0.6, "spending-by-category": 0.3}, fits=fits, choose=choose)

    d = await SkillRouter(CATALOG, judge, Settings(load_at=0.8, suggest_at=0.4)).decide(Turn("something vague"))

    assert d.load == ()
    assert d.suggest == ("visa-statement",)  # the other one does not fit enough to be offered


def test_no_more_than_max_suggest_are_offered():
    trace = Trace(
        need=0.9,
        candidates=tuple((s.name, 0.2) for s in CATALOG),
        fits={s.name: 0.8 for s in CATALOG},
        picked={s.name: 0.2 for s in CATALOG},
    )

    assert decide_from_trace(Settings(max_suggest=2), trace).suggest == ("visa-statement", "spending-by-category")


def test_a_trace_without_a_pick_is_ordered_by_the_ranking():
    trace = Trace(
        need=0.9,
        candidates=(("visa-statement", 0.7), ("spending-by-category", 0.2)),
        fits={"visa-statement": 0.92, "spending-by-category": 0.95},
    )

    assert decide_from_trace(Settings(), trace).load == ("visa-statement",)  # its fit stands in for the pick


async def test_no_more_than_max_load_the_rest_become_suggestions():
    fits = {"visa-statement": 0.95, "spending-by-category": 0.9, "subscriptions": 0.85}
    judge = scripted(
        dict.fromkeys(fits, 0.3), fits=fits, choose={"visa-statement": 0.34, "spending-by-category": 0.33, "subscriptions": 0.33}
    )

    d = await SkillRouter(CATALOG, judge, Settings(max_load=2, load_at=0.3)).decide(Turn("all of it at once"))

    assert d.load == ("visa-statement", "spending-by-category")
    assert d.suggest == ("subscriptions",)


async def test_ranking_sees_the_whole_catalog_and_verification_only_the_candidates():
    judge = scripted({"visa-statement": 0.5, "spending-by-category": 0.3, "subscriptions": 0.1, "card-limits": 0.05})

    await SkillRouter(CATALOG, judge, replace(Settings(), max_candidates=2)).decide(Turn("something"))
    _, ranking, verify = judge.calls

    assert set(ranking[1]["skill"].options) == {s.name for s in CATALOG}
    fits = {k.removeprefix("fits:") for k in verify[1] if k.startswith("fits:")}
    assert fits == {"visa-statement", "spending-by-category"}
    assert "need" not in verify[1]  # already asked in the first round


async def test_judge_gets_the_request_and_the_context_as_given():
    judge = scripted({"visa-statement": 0.9})

    await SkillRouter(CATALOG, judge).decide(Turn("and for April?", context="context from the product"))
    state = judge.calls[0][0]

    assert (state["request"], state["context"]) == ("and for April?", "context from the product")


async def test_verification_sees_the_head_of_the_skill_text():
    catalog = [skill("visa-statement", text="x" * 5000)]
    judge = scripted({"visa-statement": 1.0}, fits={"visa-statement": 0.9})

    await SkillRouter(catalog, judge, Settings(head_chars=100, skip_verify_at=None)).decide(Turn("visa"))
    question = judge.calls[2][1]["fits:visa-statement"].instructions

    assert "x" * 100 in question and "x" * 101 not in question
    assert "candidates" not in judge.calls[2][0]  # a candidate's text lives only in its own question


# --- failures do not break the turn ---------------------------------------------------------------


class Boom(ScriptedJudge):
    """A judge that fails from the given call onwards: 1 is the first round, 3 is verification."""

    def __init__(self, fail_on: int, pick: Mapping[str, float], slow: float = 0.0) -> None:
        super().__init__(scripted(pick).script)
        self.fail_on, self.slow = fail_on, slow

    async def ask(self, state, questions):
        if len(self.calls) + 1 >= self.fail_on:
            self.calls.append((state, questions))
            if self.slow:
                import asyncio

                await asyncio.sleep(self.slow)
            raise RuntimeError("provider unavailable")
        return await super().ask(state, questions)


async def test_ranking_failure_changes_nothing():
    judge = Boom(fail_on=1, pick={})

    d = await SkillRouter(CATALOG, judge).decide(Turn("something"))

    assert (d.load, d.suggest) == ((), ())
    assert "provider unavailable" in d.trace.failure


async def test_slow_judge_is_cut_by_timeout_and_changes_nothing():
    judge = Boom(fail_on=1, pick={}, slow=5)

    d = await SkillRouter(CATALOG, judge, Settings(timeout=0.05)).decide(Turn("something"))

    assert (d.load, d.suggest) == ((), ())
    assert d.trace.failure == "timeout"


async def test_the_judges_own_timeout_cuts_each_call_within_a_generous_decision_timeout():
    judge = Boom(fail_on=1, pick={}, slow=5)
    judge.timeout = 0.05

    d = await SkillRouter(CATALOG, judge, Settings(timeout=30)).decide(Turn("something"))

    assert (d.load, d.suggest) == ((), ())
    assert "did not answer within 0.05 s" in d.trace.failure
    assert d.trace.seconds < 5


def test_a_judge_timeout_that_is_not_positive_fails_at_once():
    with pytest.raises(JudgeMisconfigured, match="timeout"):
        SkillRouter(CATALOG, ScriptedJudge(lambda s, q: {}, timeout=0))


async def test_verification_failure_suggests_best_of_the_ranking():
    judge = Boom(fail_on=3, pick={"visa-statement": 0.6, "spending-by-category": 0.3, "subscriptions": 0.1})

    d = await SkillRouter(CATALOG, judge).decide(Turn("statement"))

    assert d.load == ()
    assert d.suggest == ("visa-statement", "spending-by-category", "subscriptions")


async def test_unreadable_skill_drops_out_of_candidates():
    async def broken() -> str:
        raise OSError("file is gone")

    catalog = [Skill("visa-statement", "Description", broken), *CATALOG[1:]]
    judge = scripted({"visa-statement": 0.6, "spending-by-category": 0.3}, fits={"spending-by-category": 0.9})

    d = await SkillRouter(catalog, judge).decide(Turn("statement"))

    assert d.load == ("spending-by-category",)
    assert not any(k == "fits:visa-statement" for k in judge.calls[2][1])


@pytest.mark.parametrize(
    "catalog",
    [
        [skill("dup"), skill("dup")],
        [Skill("empty", "  ", skill("x").read)],
    ],
)
def test_broken_catalog_fails_at_once(catalog):
    with pytest.raises(ValueError):
        SkillRouter(catalog, scripted({}))


# --- search behind find_skill ---------------------------------------------------------------------


async def test_empty_catalog_returns_no_search_results_without_asking_the_judge():
    judge = ScriptedJudge(lambda state, questions: {})
    router = SkillRouter([], judge)

    assert await router.search("visa statement") == []
    assert await router.search("subscriptions", limit=2) == []
    assert judge.calls == []


async def test_search_returns_best_skills_by_probability():
    judge = scripted({"subscriptions": 0.5, "dispute": 0.3, "card-limits": 0.1, "visa-statement": 0.05})

    found = await SkillRouter(CATALOG, judge).search("charged for a subscription I cancelled", limit=2)

    assert [s.name for s in found] == ["subscriptions", "dispute"]
    assert judge.calls[0][0]["request"] == "charged for a subscription I cancelled"


async def test_search_failure_returns_nothing():
    found = await SkillRouter(CATALOG, Boom(fail_on=1, pick={})).search("something")
    assert found == []


# --- the trace, and skipping verification ---------------------------------------------------------


async def test_ranking_probabilities_are_in_the_trace():
    judge = scripted({"visa-statement": 0.7, "dispute": 0.2}, fits={"visa-statement": 0.9})

    d = await SkillRouter(CATALOG, judge, Settings(max_candidates=2)).decide(Turn("visa"))

    assert d.trace.candidates == (("visa-statement", 0.7), ("dispute", 0.2))
    assert d.trace.stage == "verify"


@pytest.mark.parametrize("skip_verify_at", [0.8, None])
@pytest.mark.parametrize("max_suggest", [0, 1, 3])
async def test_zero_max_load_only_suggests_with_or_without_verification(skip_verify_at, max_suggest):
    judge = scripted({"visa-statement": 0.95}, fits={"visa-statement": 0.95})
    settings = Settings(max_load=0, max_suggest=max_suggest, skip_verify_at=skip_verify_at)

    decision = await SkillRouter([CATALOG[0]], judge, settings).decide(Turn("visa statement"))

    assert decision.load == ()
    assert decision.suggest == (("visa-statement",) if max_suggest else ())
    assert decision.trace.failure is None
    assert decide_from_trace(settings, decision.trace) == decision


async def test_a_confident_ranking_skips_verification_when_the_product_allows_it():
    judge = scripted({"visa-statement": 0.9, "dispute": 0.05})

    d = await SkillRouter(CATALOG, judge, Settings(skip_verify_at=0.8)).decide(Turn("visa"))

    assert (d.load, d.trace.stage) == (("visa-statement",), "skip")
    assert len(judge.calls) == 2  # the ranking round, and no verification


@pytest.mark.parametrize(
    ("best", "settings"),
    [
        pytest.param(0.6, Settings(skip_verify_at=0.8), id="ranking is not confident enough"),
        pytest.param(0.99, Settings(skip_verify_at=None), id="the product turned skipping off"),
    ],
)
async def test_verification_runs_otherwise(best, settings):
    judge = scripted({"visa-statement": best, "dispute": 0.3}, fits={"visa-statement": 0.9})

    await SkillRouter(CATALOG, judge, settings).decide(Turn("visa"))

    assert len(judge.calls) == 3


def test_thresholds_can_be_refitted_on_saved_traces():
    trace = Trace(candidates=(("visa-statement", 0.9), ("dispute", 0.05)), need=0.9, fits={"visa-statement": 0.3})

    assert decide_from_trace(Settings(skip_verify_at=None), trace).load == ()  # verification says no
    assert decide_from_trace(Settings(skip_verify_at=0.8), trace).load == ("visa-statement",)  # the threshold says yes


async def test_both_questions_can_be_written_for_the_product_domain():
    settings = Settings(
        need_question="Does this need a bank skill?",
        rank_question="Which bank skill fits best?",
        fits_question="Does {name} do this? {text}",
    )
    judge = scripted({"visa-statement": 0.6}, fits={"visa-statement": 0.9})

    await SkillRouter(CATALOG, judge, settings).decide(Turn("a statement"))
    need, ranking, verify = judge.calls

    assert need[1]["need"].instructions == "Does this need a bank skill?"
    assert ranking[1]["skill"].instructions == "Which bank skill fits best?"
    assert verify[1]["fits:visa-statement"].instructions.startswith("Does visa-statement do this? # visa-statement")
