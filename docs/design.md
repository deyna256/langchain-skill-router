# How Skill Router works

## The problem

An agent with tens or hundreds of skills keeps the name and description of every one of them in the
system prompt, on every model call. The list grows with the catalog, takes up context and distracts the
model: the more options there are, the worse it chooses — Anthropic reports a marked drop past 30–50
tools, and on a catalog of 182 skills an agent loaded the wrong skill in 16.8% of requests. Skill Router
decides on each turn which skills are needed, and the model sees only those, so that it works on the
user's request rather than on the catalog.

## What the measurements settled

This section records the earlier measurements that informed the design. The latest benchmark, of the
0.2.2 release, is reported in the [README](../README.md#results). What it settled, and what it left for the
next version:

- **Selection is no longer the weak point.** The loaded skill was the right one in 96% of loads, and the
  right skill reached the model in 85% of turns against 55% with the full list. Skill Router answered 90% of
  turns correctly against 88% for the full list and 88% for perfect selection — the same answers from a
  quarter of the context.
- **Skill quality is the ceiling.** An earlier run put Skill Router five points below the full list. Six skills
  in the testbed described a procedure that contradicted the rule the expected answer was computed from —
  one told the agent to look for an outgoing payment when the question was whether a counterparty had paid.
  Every variant that loads skills paid for it, perfect selection included; the full list, which rarely reads
  a skill, did not. With those six aligned, the gap reversed.
- **A wrong skill is the costly error.** Turns worked from a wrong skill were answered correctly 72% of the
  time, against 92% with the right one, which is what the load threshold of 0.9 is there to buy.
- **A rule inside a skill is what routing is for**: on the 21 turns whose answer depends on one, Skill Router
  scored 76%, the full list 67%, and an agent with no catalog at all 19%.

## Decisions

1. **Skill Router collects nothing and remembers nothing.** The catalog and the turn's context come from the
   product or its framework, and each turn is decided on its own. Nothing is carried over from the turn
   before it, so the library needs no checkpointer and no store of its own.
2. **The judge is a fast classifier that returns probabilities** (Jev today). The logic — which
   questions to ask and how to read the answers — stays in Skill Router; a provider is plugged in through
   the `Judge` port, which speaks in "pick one" and "yes or no".
3. **A decision is the set of skills for the turn:** load (the instruction text goes straight into the
   request) and suggest (two or three candidates the model chooses from).
4. **Every threshold is a product setting** (`Settings`), fitted on the product's own data.
5. **A Skill Router failure does not break the conversation:** the worst outcome is that the agent gets its
   skills the ordinary way.

## The turn

When the catalog is empty, `decide()` returns an empty decision with `trace.stage == "empty"`,
and `search()` returns an empty list. Neither calls the judge. The `empty` stage distinguishes a
successful empty-catalog decision from a failure, whose stage is blank and whose `failure` records the reason.

```
user message
  │
  ├─ 1. "Is a skill needed at all" (~400 tokens), asked alongside the ranking rather than before it
  │     └─ answered low → the decision is ready, and nothing is loaded
  │
  ├─ 2. Ranking over the catalog: pick by name and description → the best candidates with probabilities
  │     (over the provider's limit → parts in parallel, then a merge)
  │     └─ the first candidate at or above `skip_verify_at` (0.9) → load it without verifying
  │
  └─ 3. Verification, one call over the heads of the candidates' texts:
        "which of these is the right one?" — a pick with the texts side by side — settles which skill,
        and "do these instructions do what the request asks?" for each on its own admits a candidate at all
        → the pick, if verification is sure of it (`load_at`), is loaded; otherwise up to `max_suggest`
          admitted candidates are offered; none admitted → nothing
```

Setting `max_load=0` disables loading, including when confident ranking skips verification.
The selected candidate is offered instead, subject to `max_suggest`.

**Why it is shaped this way**, from the testbed measurements:

- Verification asks two different things. Ordered by their independent yes/no answers alone, lookalikes
  tie (0.8 against 0.9, say) and the wrong one often comes first: over 263 conversation turns on the bank
  testbed the right skill was first by `fits` in 67% of turns, against 92% by the ranking over
  descriptions, and loading the right skill together with a neighbour dropped the answer from 91% correct
  to 76%. So a pick over the candidates' texts decides which skill is loaded, one by default, and `fits`
  only admits candidates — the split TypeSafe's skill-suggestion cookbook uses as well.
- How sure the pick is decides between loading and offering. At 0.9 and above the pick was right in 90% of
  requests, and a wrong skill loaded costs more than none: turns with a wrong skill were answered correctly
  in 71% of cases, with nothing loaded in 84%, with the right skill in 89%. Below that the model gets a
  short list to choose from. Short, because offered four candidates the model read the right one as often
  as from two, but answered worse: 82% against 92%.
- A ranking sure of its first candidate is rarely overturned by verification, so at 0.9 the second call is
  skipped: that spared it in 44% of requests with the same share of right loads.
- A candidate's text goes in its own question rather than in the shared state, because otherwise its
  score depends on its neighbours in the same call — a shift of up to 0.46, against no more than 0.05
  this way — and the size of a call stops growing with the number of candidates.
- "Is a skill needed at all" travels with the ranking instead of preceding it. Asked first it ends
  about one turn in fifty on its own and costs every other turn a round trip; asked alongside, it costs
  neither. At a threshold of 0.05 it errs towards "not needed" in 0-0.7% of requests.

## The interface

The core is imported from the package root, and the parts that carry a dependency from their own
modules:

```python
from langchain_skill_router import Settings, SkillRouter, Turn  # core: no framework, no provider
from langchain_skill_router.langchain import SkillRouterMiddleware  # pulls in langchain and deepagents
from langchain_skill_router.providers.jev import JevJudge  # pulls in typesafe-sdk

router = SkillRouter(catalog, judge, settings)
decision = await router.decide(Turn(request, context))  # load / suggest + trace
found = await router.search(query)  # for the find_skill tool
```

`Settings` carries three kinds of knob. **How much:** `max_candidates`, `max_load`, `max_suggest`,
`head_chars`, `request_chars`, `context_chars`, `budget_share`, `timeout`. **Thresholds:** `need_at`,
`load_at`, `suggest_at` and `skip_verify_at`, which `None` turns off. **The questions the judge is asked:**
`need_question`, `rank_question`, `pick_question` and `fits_question` — the defaults are written for a general
assistant, and wording that names the product's own domain separates better. `Trace` carries the
ranking probabilities, the answer to every question, where the decision ended, how long it took and
what failed.

`Settings.timeout` limits a whole decision, which may take several calls to the judge. A judge may declare
`timeout`, the limit on one call: the router enforces it on every call whatever the adapter, and the adapter
passes it to its client so the provider's own default does not cut in first (`JevJudge(timeout=...)`; left
out, the SDK's 10 s).

Each threshold is compared against the trace field of the same name: `need_at` against `trace.need`,
`load_at` against verification's pick in `trace.picked`, `suggest_at` against each candidate's
`trace.fits`, `skip_verify_at` against the first of `trace.candidates`. Candidates are ordered by the pick;
a trace recorded without one follows the ranking, and its fit stands in for the pick.

## Plugging in a different judge

`Judge` is declared in the core, because the router is its consumer: an adapter satisfies the
interface, and the interface does not follow an adapter. What an adapter has to provide:

| requirement | why the router needs it |
|---|---|
| answer every question of a call in **one** request | this is where the cost saving comes from; a provider that answers one at a time fans out inside the adapter |
| a `Pick` returns a probability for **every** option | the router ranks candidates from that distribution, not from the winner alone |
| probabilities are calibrated and within [0, 1] | every setting is a threshold they are compared against |
| declare `limits` | the router splits a catalog that does not fit one call; `Limits()` says the provider has none |
| raise `JudgeUnavailable` or `JudgeMisconfigured` | the first falls back to the full catalog, the second surfaces |

The split matters. Rejected credentials are misconfiguration, because no retry fixes them and a
fallback would hide a broken setup for the life of the process. A rejected request is not, because the
router already answers that by splitting the catalog and asking again.

None of this is left to be read carefully:

```python
from langchain_skill_router.testing import check_judge


async def test_my_adapter():
    await check_judge(MyJudge())
```

One real call checks the shape of the answers and that they are judgements rather than numbers: an
obvious yes has to outscore an obvious no, and the right option has to win the pick.

**Provider limits.** An adapter declares `Limits` — for Jev, 32k tokens for the state plus the longest
question, and 255 options. Skill Router estimates the size of every call in advance, splits the catalog into
parts that fit, truncates the request and the context, and rejects settings that cannot fit at all. If
a part is refused anyway, it is asked again in halves.

## Inside the agent (deepagents)

`SkillRouterMiddleware` takes the place of `SkillsMiddleware`, under the same name. Skills are still
discovered by the ordinary middleware through `state["skills_metadata"]`; on a new user message Skill Router
decides, and the model sees only the picked skills. The layout follows the provider's prompt cache, which
serves a new request only as far as it repeats an earlier one: the system message gets a section that is
the same on every call, and the turn's skills — the loaded one's instructions, the others by name and
description — go in a message right after the user's request, written into the conversation. So every call
of the turn, and the next turn's first call, begins with everything sent before. Kept in the request alone,
the message vanished on the next turn and took the conversation's cache with it: on the testbed the first
call of a turn got the conversation from the cache in 16–24% of turns, against 64% without the message. The
message is marked (`is_skill_message`), is never taken for the user's request and stays out of the judge's
context. On failure the request goes to the ordinary middleware with the full list.

The middleware relies only on public names of deepagents and langchain. Its state extends the one
`SkillsMiddleware.state_schema` declares, so it follows whatever the installed deepagents accepts in
`skills_metadata`, including `None` for a catalog that is not loaded yet. It adds to the system message and
lists skills with helpers of its own.

The middleware uses the user message's text for the judge's request. Image and file attachments stay
in the agent's conversation but are not included in the routing request.

The judge sees the turn's context as the product assembles it; the default, `recent_context`, is the
last few things the user asked and the agent answered, without tool calls and their results, which would
otherwise crowd the previous request out of the window. `find_skill(query)` covers the case where the model needs a skill that
is not in the list. The optimisation applies when the agent is run asynchronously.

## Known limits

- **A skill loaded again is written again.** The conversation keeps every turn's skill message, and a
  skill picked on consecutive turns appears in each of them. Referring back to the earlier copy would need
  the "already loaded" knowledge the limit below explains the library does not rely on: summarization can
  replace that turn in what the model sees while the state still holds it. The repeat is read from the cache
  on later turns, so it costs context rather than recomputation.
- **Two seconds is out of reach on a large catalog.** On 236 skills the catalog splits into three
  parts, so a full pass costs about 58k tokens and about 3 s; only the cheap decisions land inside two
  seconds. The size estimate is part of the problem: 0.8 tokens per character is assumed against about
  0.68 measured, which causes unnecessary splitting.
- **Thresholds have to be fitted per product**, and there is no procedure in the library for it yet.
  `decide_from_trace` is kept separate from the calls so traces can be refitted without paying the provider
  again, but the workflow around that is missing.
- **Every turn pays for a full pass.** Skill Router deliberately carries nothing between turns, so a
  continuation of a topic costs the same as a new one: 3.1 s against the 1.6 s it cost while the
  library kept track of what was loaded. Removing that also cost about three points of skill-choice
  accuracy on continuations, measured by replaying the recorded answers.

  It was removed because it could not be made reliable. Graph state does not survive an invocation
  without a checkpointer, a private field cannot even be passed back in, and the message history is no
  better because middleware clips and offloads tool results. The errors are asymmetric: believing a
  skill is loaded when it is not makes the agent answer without the procedure, silently, while not
  knowing about a loaded skill only costs a pass over the catalog. Bringing it back needs a carrier the
  library is willing to ask an application for.

Open work is tracked in issues.

## How this differs from what already exists

| approach | how it chooses | what is missing |
|---|---|---|
| progressive disclosure (Agent Skills, the deepagents `SkillsMiddleware`) | every skill's name and description in the prompt, the text on request | the list grows with the catalog, and the choice is still the model's |
| tool search / `defer_loading` ([Anthropic](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)) | the model searches for itself with regex or BM25 | the model has to realise it should search; an extra step |
| `LLMToolSelectorMiddleware` ([LangChain](https://docs.langchain.com/oss/python/langchain/middleware/built-in)) | a separate LLM call on every step | slow and expensive |
| vector retrieval ([RAG-MCP](https://arxiv.org/abs/2505.03275), [dynamic-tools](https://github.com/RauhanAhmed/langchain-dynamic-tools-middleware)) | similarity between descriptions | no answer to "is a skill needed at all", no confidence; ordinary retrievers are poor at finding tools ([ToolRet](https://aclanthology.org/2025.findings-acl.1258/)) |
| skill suggestion on Jev ([TypeSafe](https://docs.typesafe.ai/cookbooks/skill_suggestion.md)) | a classifier plus candidate verification | the whole catalog stays in the prompt |
