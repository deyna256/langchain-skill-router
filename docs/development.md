# Development guide

These are Skill Router's coding rules. What the product does is in the [README](../README.md) and
[design](design.md); commands are in the [Justfile](../Justfile). LLM instructions should link here
instead of copying these rules.

## Layers and the dependency rule

The package has four parts and dependencies point inward:

| part | may import |
|---|---|
| `core/` — types, judge port, router | the standard library only |
| `providers/` — judge adapters | `core`, its own provider SDK |
| `langchain/` — agent middleware | `core`, langchain, deepagents |
| `testing/` — conformance kit and fakes | `core` |

Nothing in `core/` may import langchain, deepagents or a provider SDK. An import pointing outward is a
layering break and blocks a review.

The `Judge` interface belongs to `core`, because the router is its consumer. Adapters satisfy the
interface; the interface does not follow an adapter.

## Public API

`langchain_skill_router.__all__` is the public API. Everything else is an implementation detail and may
change without notice. Adding a name to `__all__` is a commitment; removing one needs a deprecation
release after 1.0.

`__init__.py` exports the core only. The middleware and each provider adapter are imported from their
own modules, so that importing the package does not pull in a framework, and so that an optional
dependency never becomes mandatory through a re-export.

## Imports

Use absolute imports (`from langchain_skill_router.core.types import Settings`), never relative ones. Import
an optional dependency inside the module that needs it, and raise an error that names the extra to
install when it is missing.

## Data and types

Public data is a frozen dataclass: `Skill`, `Turn`, `Settings`, `Decision`, `Trace`. Frozen means a
value can be shared without copying and a default instance is safe as a module-level singleton, which
is what `DEFAULTS` is.

Annotate every public function and method. Accept the narrowest type that works (`Sequence`, `Mapping`)
and return a concrete one (`tuple`, `dict`). Use `tuple` for sequences that belong to a frozen value.

Validate what the product supplies at construction, not at use: duplicate skill names, empty
descriptions and settings that cannot fit the provider's limits all raise from `SkillRouter.__init__`.
A failure at construction is a bug report; the same failure during a turn is a silent fallback.

## Settings and thresholds

Every number a product might want to change lives in `Settings`, with a comment saying what it does.
Do not hardcode a threshold in the router. Thresholds are compared against calibrated probabilities, so
a judge that cannot produce them cannot be used.

Keep policy separate from calls: `decide_from_trace` turns a `Trace` into a `Decision` without touching the
judge, so thresholds can be refitted on recorded traces without spending money.

## Asynchronous code

The decision path is asynchronous end to end. Use `asyncio.timeout` for the whole decision, with the
judge's own `timeout` enforced on each call inside it, and `asyncio.gather` for work that is genuinely
independent — catalog parts, skill reads, candidate checks.

Never block the event loop: no synchronous HTTP, no file reads outside the backend, no `time.sleep`.
Measure elapsed time with `time.monotonic`.

Keep per-turn data in local variables. The router and the middleware are shared between concurrent
turns, so anything stored on `self` must be derived from the catalog, not from a turn.

## Failure

Two judge failures, with different handling:

- `JudgeUnavailable` — network, timeout, provider error. Fall back silently to the full catalog and
  record the reason in `Trace.failure`.
- `JudgeMisconfigured` — missing or rejected credentials, an adapter that breaks the contract. Never
  swallowed: it surfaces from `SkillRouter`, and through the middleware on the first turn, where the
  router is built from the turn's catalog.

An adapter decides which is which, and the rule is whether a retry could help. Credentials the provider
rejects are misconfiguration. A rejected request, including one that is too large, is not: the router
answers that by splitting the catalog and asking again, so classifying it as permanent would break a
path that already works.

Never let a Skill Router failure break the agent. The fallback is always the same: the agent gets the full
catalog, exactly as it would without Skill Router.

If selected instructions cannot be read (`OSError`, including timeouts, or `UnicodeError`), the
middleware warns and uses the ordinary full catalog for that model call. No partially loaded text is
injected. The next model call may try reading again. This recovery wraps instruction reads only:
model errors, backend programming errors and cancellation propagate. `on_decision` reports the routing
decision before instructions are loaded; it does not report this later fallback.

Catch narrowly and say what failed. A bare `except Exception` is acceptable only at the outermost
boundary of a decision, and it must record what it caught.

An exception from a user callback, such as `on_usage`, must not discard a successful answer. Log
the exception type only, not its message or traceback, which may contain private data.
Cancellation still propagates.

## Logging

One logger per module, `logging.getLogger(__name__)`, so an application configures the single name
`langchain_skill_router`. The library never adds handlers, sets levels or calls `basicConfig`, and never
accepts a logger argument.

| level | content |
|---|---|
| `DEBUG` | the decision trace: candidates with probabilities, stage, timings, catalog parts |
| `WARNING` | judge unavailable, fell back to the full catalog; timeout; a user callback such as `on_usage` raised |
| `ERROR` | misconfiguration, once per process |
| `INFO` | nothing |

The request and the context are the end user's own text and may contain personal data. They never
appear above `DEBUG`. Warnings and errors carry skill names, probabilities, timings and exception
type names only.

`on_decision` is the programmatic hook for metrics and benchmarks. Logging is for people; do not use
one in place of the other.

## Names, comments and length

Identifiers are English. Comments explain why, not what: a line that restates the code is noise, and a
line that records a measurement or a constraint is worth keeping. Note the source of a constant that
came from a measurement.

Lines are at most 135 characters, which suits one thought per line. `ruff format` decides layout; do
not fight it.

Docstrings are for modules, public classes and anything whose contract is not obvious from the
signature. A module docstring says what the module is for and how it fits the layers.

## Tests

Test behaviour through public interfaces. The router is exercised through `decide` and `search`, the
middleware through an agent built with a fake model and a fake judge. Do not test private helpers
directly and do not assert on call counts inside the router.

`langchain_skill_router.testing.ScriptedJudge` is the scripted judge, and it ships in the package because
anyone testing an agent that uses Skill Router needs one too. A test states the answers the judge gives and
asserts on the resulting `Decision`; that keeps tests independent of how many calls the router makes to
get them.

A new adapter is checked against the contract with `langchain_skill_router.testing.check_judge`. It makes
one real call, so it belongs wherever the keys are — for us, in the testbed, not here.

Cover the failure paths as deliberately as the success path: the judge raising, timing out, returning a
malformed answer, and the catalog exceeding the provider's limits.

Tests never call a provider. There are no live tests in this repository.

## Tooling

`uv` manages the environment and the lock file. `ruff` lints and formats, configured in
`pyproject.toml`. `ty` checks types with the optional dependencies installed, so provider adapters
resolve.
All of them run from the [Justfile](../Justfile) and in CI.

Add a dependency only when it carries real weight, and prefer the standard library. A new runtime
dependency needs a reason in the pull request; a new optional dependency needs an extra.

## Review checklist

- Do all imports point inward? Does anything in `core/` reach for a framework or a provider?
- Is a new public name in `__all__` deliberate, and is it documented?
- Is every new threshold in `Settings` with a comment?
- Does the change keep the fallback intact — can a judge failure still not break the agent?
- Are the failure paths tested, not only the success path?
- Does any log line above `DEBUG` carry user text?
- Does the changelog entry describe the change from the user's side?
- Is the documentation for the changed behaviour updated in the same change?
