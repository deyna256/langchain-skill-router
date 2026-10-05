<div align="center">

<img src="https://raw.githubusercontent.com/deyna256/langchain-skill-router/main/docs/assets/banner.svg" alt="Skill Router: per-turn skill selection for LangChain deepagents, with a pluggable judge" width="100%">

<h3>Per-turn skill routing for LangChain deepagents</h3>

<p>A drop-in replacement for the <a href="https://github.com/langchain-ai/deepagents">deepagents</a>
<code>SkillsMiddleware</code>: on each user turn a fast judge decides which <code>SKILL.md</code> skills are
needed, and only those are loaded, so a catalog of hundreds stays out of the prompt. Bring any judge: a
hosted model, a self-hosted one, or plain rules. An adapter for
<a href="https://docs.typesafe.ai/introduction">Jev</a> is included.</p>

<p><strong>113.0k&nbsp;&rarr;&nbsp;25.8k input tokens per turn</strong> on a 236-skill catalog, and the agent
answered <strong>90%</strong> of questions correctly against <strong>88%</strong> with the whole catalog in
the prompt. <a href="#results">See the benchmark&nbsp;&rarr;</a></p>

[![CI](https://github.com/deyna256/langchain-skill-router/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/deyna256/langchain-skill-router/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-skill-router?label=pypi)](https://pypi.org/project/langchain-skill-router/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-skill-router?label=python)](https://pypi.org/project/langchain-skill-router/)
[![License: MIT](https://img.shields.io/github/license/deyna256/langchain-skill-router)](https://github.com/deyna256/langchain-skill-router/blob/main/LICENSE)
[![Judge: pluggable](https://img.shields.io/badge/judge-pluggable-4b32c3)](#bring-your-own-judge)
<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

[Quick start](#quick-start) · [How it works](#how-it-works) · [Your own judge](#bring-your-own-judge) · [Results](#results) · [FAQ](#faq) · [Docs](https://github.com/deyna256/langchain-skill-router/blob/main/docs/design.md)

</div>

---

## Why Skill Router

[deepagents](https://github.com/langchain-ai/deepagents) lists every skill's name and description in
the system prompt on every model call. With a handful of skills that is fine. With hundreds, the list
takes tens of thousands of tokens per call, and the model has to pick the right procedure from a crowd
of similar ones.

Skill Router replaces the built-in `SkillsMiddleware` with one that decides **per user turn**:

- **Only what the turn needs.** A confident pick is loaded with its instructions. When the pick is unsure,
  the model gets a short list of up to three candidates to choose from, and it can search the rest of the
  catalog with the `find_skill` tool.
- **Any judge, by design.** The judge is a small protocol: typed questions in, probabilities out. Use the
  included Jev adapter, a self-hosted inference model, deterministic rules or anything else that can answer.
  The routing core depends on neither LangChain nor any provider.
- **Calibrated, not guessed.** Every question is answered with a probability, so "load it", "offer it"
  and "skip it" are thresholds you can read and tune. They are not buried in a prompt.
- **Conversation-aware.** The judge also sees the recent conversation, so a follow-up like "and for
  April?" still routes to the skill the thread is about.
- **Safe by default.** A timeout or an outage of the judge gives the agent the full catalog, exactly as it
  would be without Skill Router. Bad credentials raise instead of hiding a broken setup.
- **Cache-friendly.** The system prompt is identical on every call, and the turn's skills are written after
  the user's message. The provider's prompt cache keeps working across turns.

## Quick start

```sh
pip install "langchain-skill-router[jev]"   # with the Jev adapter
pip install langchain-skill-router          # with your own judge
```

Put skills in `./skills/<name>/SKILL.md`, with `name` and `description` in YAML front matter (the Agent
Skills format deepagents uses). Set `TYPESAFE_API_KEY` ([get a key](https://console.typesafe.ai/settings/keys))
and your model provider's key. Jev is the judge in this example; see [below](#bring-your-own-judge) for
your own.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from langchain_skill_router.langchain import SkillRouterMiddleware
from langchain_skill_router.providers.jev import JevJudge

backend = FilesystemBackend(root_dir=".", virtual_mode=True)
skill_router = SkillRouterMiddleware(backend=backend, sources=["/skills/"], judge=JevJudge())

agent = create_deep_agent(
    model="anthropic:claude-sonnet-5",
    backend=backend,
    skills=["/skills/"],
    middleware=[skill_router],  # takes the place of the built-in SkillsMiddleware
)
await agent.ainvoke({"messages": [{"role": "user", "content": "I need a statement for the embassy"}]})
```

Selection runs on `ainvoke` and `astream`. A synchronous run falls back to the ordinary skills
middleware. Requires Python 3.11+.

## How it works

<img src="https://raw.githubusercontent.com/deyna256/langchain-skill-router/main/docs/assets/how-it-works.svg" alt="A user turn is ranked with a need gate, then verified; the skill is loaded, offered in a short list, or nothing is loaded. A pluggable judge answers every question; on failure the agent gets the full catalog." width="100%">

On each new user message, Skill Router makes one decision, and the rest of the turn's model calls reuse it:

1. **Rank and gate, in parallel.** The judge ranks the catalog by description against the request and the
   recent conversation. In the same round it answers whether the request needs a skill at all. A catalog
   larger than the judge's declared limits (for Jev, 32k tokens and 255 options per call) is split into
   parts, keeping related skills together, and the part winners are ranked again.
2. **Verify.** The judge reads the start of each top candidate's `SKILL.md`, picks one and checks whether
   each candidate does what the user asked. A ranking that is already sure skips this second call.
3. **Load or offer.** A pick verified at 0.9 or higher is loaded with its instructions. Otherwise the model
   is offered a short list. If no skill is needed, nothing is loaded.

A decision is capped at two seconds by default (`Settings.timeout`). The routing core has no framework
dependency: `SkillRouter` works on any list of skills.

## Bring your own judge

A judge answers two kinds of question in one call: `Pick` (a probability for every option) and `YesNo`
(a probability of yes). It declares its per-call `limits`, which the router uses to split a large catalog,
and optionally a per-call `timeout`:

```python
from langchain_skill_router import Answer, Limits, Pick, YesNo


class MyJudge:
    limits = Limits(max_tokens=8_000, max_options=100)  # what one call can take; Limits() for no limit
    timeout = 5.0  # seconds per call, enforced by the router; None for no limit

    async def ask(self, state, questions):
        # state: {"request": ..., "context": ...}; questions: {key: Pick | YesNo}
        # Call a self-hosted model, a classifier or rules here, and answer every key.
        return {
            key: Answer({option: 1 / len(q.options) for option in q.options}) if isinstance(q, Pick) else Answer({"yes": 0.5})
            for key, q in questions.items()
        }
```

Pass it as `SkillRouterMiddleware(..., judge=MyJudge())`. `langchain_skill_router.testing.check_judge` checks
an adapter against the contract, and `ScriptedJudge` answers from a script in your tests. The probabilities
are compared against thresholds, so the closer they are to calibrated, the better the defaults fit.

## Results

Benchmark of **version 0.2.2** on a bank-statement assistant built with deepagents: 236 skills,
**55 conversations × 5 turns** per variant (275 turns each), Jev as the judge, one agent model for all
variants. "Perfect selection" always loads the skill the question was written for: the ceiling for any router.

<img src="https://raw.githubusercontent.com/deyna256/langchain-skill-router/main/docs/assets/results.svg" alt="Input tokens per turn: 113.0k with the full catalog against 25.8k with Skill Router, 4.4 times less. Right skill in front of the model: 55% against 85%. Correct answers: 88% against 90%." width="100%">

| Metric | Skill Router | Full catalog | Perfect selection |
|---|---|---|---|
| Input tokens per turn | **25.8k** | 113.0k | 26.8k |
| Skills in the prompt, characters per call | **2.6k** | 89.2k | 2.7k |
| Right skill in front of the model | **85%** | 55% | 97% |
| Loaded skill was the right one | **96%** (230 of 239) | — | 100% |
| Input not served from the prompt cache, tokens per turn | 6.8k | 7.1k | 6.4k |
| Correct answers | **90%** | 88% | 88% |
| Answers that depend on a rule inside a skill (21 turns) | **76%** | 67% | 62% |

- **4.4× less context per turn**, with the right skill in front of the model far more often.
- **Accuracy holds.** +1.5 points against the full catalog (95% interval −1.5 to +4.7, sign test p = 0.63):
  the same answers from a fraction of the prompt. Without any skills the agent scored 82%, so the catalog
  does matter — it just does not have to be in the prompt.
- **Where a skill carries a rule the model cannot infer, routing wins**: 76% against 67% for the full list.
- **A wrong skill costs the most.** In the 18 turns where the model worked from a wrong skill, 72% of
  answers were correct, against 92% with the right one. Raise `load_at` if your catalog has many
  near-duplicate skills.
- **The prompt cache keeps working.** The provider processed 6.8k new input tokens per turn, against 7.1k
  with the full catalog (paired over 272 turns: −0.3k, 95% interval −1.2k to +0.7k). Routing shrinks the cached
  part of the prompt; it does not add work the cache used to save. The share served from the cache is lower
  (74% against 94%) only because the prompt is smaller.

An earlier run of the same benchmark put Skill Router 5 points *below* the full catalog. The difference was six
skills in the testbed catalog whose instructions contradicted the rule the expected answer was computed
from; they dragged down every variant that loads skills, including perfect selection. Skill quality is the
ceiling of any router.

Generated data, one judge and one agent model. Fit the thresholds to your own data.
[Methodology, earlier measurements and known limits →](https://github.com/deyna256/langchain-skill-router/blob/main/docs/design.md)

## Configuration

Everything is in `Settings`, passed as `SkillRouterMiddleware(..., settings=Settings(...))`:

| Knob | Default | What it does |
|---|---|---|
| `load_at` | 0.9 | How sure verification must be of its pick to load it |
| `max_suggest` | 3 | How many candidates may be offered when the pick is unsure |
| `need_at` | 0.3 | Below this "is a skill needed" probability, nothing is loaded |
| `skip_verify_at` | 0.9 | Ranking this sure skips verification (`None` turns it off) |
| `timeout` | 2.0 | Seconds for the whole decision, after which the full catalog is used |
| `need_question`, `rank_question`, … | general wording | The questions the judge is asked. Naming your domain separates better |

`JevJudge(timeout=...)` limits each call to Jev. `on_decision=` on the middleware receives every
decision's trace (probabilities, stage, timing) for logs and metrics.

## FAQ

**Does Skill Router work without deepagents?**
Yes. `langchain_skill_router` (the core) has no framework or provider dependency:
`SkillRouter(catalog, judge).decide(Turn(request, context))` returns what to load and what to suggest.

**Do I need Jev?**
No. Jev is the included adapter and what Skill Router was measured with, but any `Judge` works: a self-hosted
inference model behind your own adapter, deterministic rules, or another provider. See
[Bring your own judge](#bring-your-own-judge).

**What happens if the judge is slow or down?**
The turn gets the full catalog, as if Skill Router were not installed, and the trace records why.

**What does the judge see?**
The request and the recent conversation (the user's messages and the agent's replies, without tool
output) are sent, capped at `request_chars` and `context_chars`. So are skill names, descriptions and the
start of the candidates' instructions. Supply your own `context=` function to send less, or a
self-hosted judge to keep everything in your network.

**Does it keep state between turns?**
No. Every turn is decided from scratch. No checkpointer or extra storage is required.

## Documentation

| Read | Covers |
|---|---|
| [How it works](https://github.com/deyna256/langchain-skill-router/blob/main/docs/design.md) | Selection flow, judge interface, settings, measurements and known limits |
| [Development guide](https://github.com/deyna256/langchain-skill-router/blob/main/docs/development.md) | Architecture, conventions and testing |
| [Contributing](https://github.com/deyna256/langchain-skill-router/blob/main/CONTRIBUTING.md) | Local setup, checks, issues and pull requests |
| [Changelog](https://github.com/deyna256/langchain-skill-router/blob/main/CHANGELOG.md) | Release history |

The public API may change in minor releases while the version is `0.x`.

## Acknowledgements

Skill Router was inspired by [Jev](https://docs.typesafe.ai/introduction), TypeSafe AI's model for typed
decisions, and its first round follows TypeSafe's
[skill suggestion cookbook](https://docs.typesafe.ai/cookbooks/skill_suggestion). Skill Router is an independent
open-source project, not affiliated with or endorsed by TypeSafe AI.

## License

[MIT](https://github.com/deyna256/langchain-skill-router/blob/main/LICENSE) © 2026 Ivan Deyna
