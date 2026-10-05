"""The data the router exchanges with the product. Knows nothing about providers or frameworks."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skill:
    """A skill from the product's catalog. Skill Router does not discover skills; it is handed them."""

    name: str
    description: str
    read: Callable[[], Awaitable[str]]  # the full instructions; called for candidates only
    group: str | None = None


@dataclass(frozen=True)
class Turn:
    """One turn of a conversation: everything a decision needs. The product assembles the context.

    A turn carries no memory of the ones before it. Whether a skill was loaded earlier cannot be known
    reliably — graph state does not survive an invocation without a checkpointer, and a `read_file` call
    stays visible in the history after middleware has clipped the text it fetched. Getting that wrong in
    the "already loaded" direction would leave the agent answering without the procedure, silently.
    """

    request: str
    context: str = ""


@dataclass(frozen=True)
class Settings:
    """Everything a product may want to tune."""

    max_candidates: int = 6  # how many candidates ranking passes to verification
    # One skill by default. Two lookalikes loaded side by side argue with each other: on the bank testbed a turn
    # with the right skill alone was answered correctly in 91% of cases, with the right skill and a neighbour
    # in 76% — no better than an agent with no skills at all.
    max_load: int = 1  # how many skills may be loaded in one turn
    # A short list. On the testbed the model read the right skill from four offered (0.2.0) as often as from
    # two (0.1.0), but answered worse: 82% against 92% of turns where the right skill was only offered.
    max_suggest: int = 3  # how many candidates may be offered for the model to choose from
    # Loading is decided by how sure verification is of its pick. At 0.9 and above the pick was right in 90%
    # of requests on the testbed, and a wrong skill loaded costs more than none: 71% of such turns were
    # answered correctly, against 84% with nothing loaded and 89% with the right skill.
    load_at: float = 0.9  # verification this sure of its pick (of the fit, with one candidate): load it
    suggest_at: float = 0.2  # "fits" at or above this: the candidate may be loaded or offered
    need_at: float = 0.3  # "a skill is needed at all" below this: load nothing
    # Ranking this sure is rarely overturned by verification: skipping it at 0.9 spared the second call in 44%
    # of requests on the testbed, with right loads at 45% either way and wrong ones at 5.7% against 5.1%.
    skip_verify_at: float | None = 0.9  # ranking this sure of its first candidate: load without verifying
    head_chars: int = 1500  # how much of a skill's text verification sees
    timeout: float = 2.0  # seconds for the whole decision
    request_chars: int = 2000  # how much of the request the judge sees (a user may paste a whole statement)
    context_chars: int = 4000  # how much of the context the judge sees, counted from the end
    budget_share: float = 0.85  # share of the provider's limit to use, leaving room for estimation error
    # Both questions the judge is asked. The defaults are written for a general assistant; wording that
    # names the product's own domain separates better (measured on bank requests, the need question went
    # from 0.80-0.97 against 0.01-0.02). The judge sees `request` and `context` in the state it is given.
    need_question: str = (
        "Does answering the user's request need a specialised skill (a documented procedure or work on the "
        "user's accounts, statements or documents) rather than a plain conversational reply?"
    )
    rank_question: str = "Which skill's instructions would help the assistant answer the user's request best?"
    # `fits_question` is formatted with the candidate's `name` and the head of its `text`.
    fits_question: str = (
        'Do these skill instructions do what the user asks for in `request`?\n\n<skill name="{name}">\n{text}\n</skill>'
    )
    # Verification picks among the candidates with their texts side by side; `fits` then only settles whether
    # any of them is right. Ordered by `fits` alone, lookalikes tie (0.8 against 0.9) and the right skill came
    # first in 67% of turns, against 92% for the ranking over descriptions. TypeSafe's skill-suggestion
    # cookbook asks the same two questions: the pick settles which, `fits` settles whether.
    pick_question: str = (
        "Exactly one of these skills is the right one to load for the user's request. Which one? "
        "Read what each actually does, not just its name."
    )


@dataclass(frozen=True)
class Trace:
    """What was asked, what came back and what broke. Recorded for measurement and debugging."""

    candidates: tuple[tuple[str, float], ...] = ()  # the best of the ranking, with probabilities, best first
    need: float | None = None
    fits: Mapping[str, float] = field(default_factory=dict)  # per candidate: does it do what the request asks
    picked: Mapping[str, float] = field(default_factory=dict)  # verification's pick among the candidates
    failure: str | None = None
    seconds: float = 0.0
    parts: int = 1  # how many parts the catalog was split into for ranking
    stage: str = ""  # where the decision ended: empty (no catalog), gate (no skill needed), skip (no verification) or verify


@dataclass(frozen=True)
class Decision:
    """What to do on this turn."""

    load: tuple[str, ...] = ()  # instructions go straight into the request
    suggest: tuple[str, ...] = ()  # listed as candidates for the model to choose from
    trace: Trace = field(default_factory=Trace)


DEFAULTS = Settings()  # Settings is frozen, so one shared instance is enough
