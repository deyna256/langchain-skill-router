# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html). While the version is `0.x` the public API
may change in a minor release.

Add the entry for a change in the same commit as the change itself.

## [Unreleased]

### Changed

- The deepagents middleware no longer uses private parts of deepagents or langchain, so a release that
  changes their internals does not break it. Its state now follows what the installed deepagents accepts
  in `skills_metadata`. Suggested skills and `find_skill` results also show a skill's allowed tools, as
  the full list does.
- The README reports the prompt cache as input tokens per turn not served from it, the same for Skill Router
  and the full catalog, instead of the share served, which was lower only because the prompt is smaller.

### Fixed

- Respect `max_load=0` when confident ranking skips verification: offer the candidate instead (#12).
- Keep the selected decision when the `on_decision` callback raises; warn once per middleware and continue the turn.
- An empty skill catalog now returns an empty decision or search results without calling the judge.
  Empty decisions use the `empty` trace stage to distinguish them from failures.
- Keep Jev's answer when the `on_usage` callback raises an exception. Warn once per judge with the
  exception type without exposing the callback's error text. Continue reporting usage on every call.
- Route messages with image or file attachments using their text, so attachment data does not crowd
  the user's request out of the judge's input.

## [0.3.0] - 2026-09-23

### Changed

- **Renamed from `langchain-loadout` to `langchain-skill-router`**, so the name says what the library does.
  This breaks every import. To migrate:

  | Before | After |
  | --- | --- |
  | `pip install langchain-loadout` | `pip install langchain-skill-router` |
  | `import langchain_loadout` | `import langchain_skill_router` |
  | `LoadoutSkillsMiddleware` | `SkillRouterMiddleware` |
  | `LoadoutError` | `SkillRouterError` |
  | `LoadoutState` | `SkillRouterState` |
  | state keys `loadout_turn`, `loadout_failed` | `skill_router_turn`, `skill_router_failed` |
  | `additional_kwargs["loadout"]` on skill messages | `additional_kwargs["skill_router"]` |

- Package metadata: a description that names the pluggable judge, keywords, classifiers and project URLs,
  so the package is easier to find on PyPI.

## [0.2.2] - 2026-09-22

### Added

- `Judge.timeout`: a judge may declare the limit on one call, in seconds. The router enforces it on every
  call, whatever the adapter, and reports a cut call as unavailable; `Settings.timeout` still limits the
  whole decision. `JevJudge(timeout=...)` passes it into every request, over the SDK's 10 s, which could
  only be changed by building the client by hand. `ScriptedJudge` and `check_judge` know it too.

### Changed

- The turn's skill message is written into the conversation instead of being added to the model request
  alone. Removed on the next turn, it stopped any earlier request from being a prefix of the new one, and
  the provider's cache lost the conversation: on the bank testbed the first call of a turn got it from the
  cache in 16–24% of turns, against 64% without the message. The message is marked (`is_skill_message`),
  never taken for the user's request and kept out of the judge's context.

## [0.2.1] - 2026-09-22

### Added

- `Settings.max_suggest`, 3 by default: at most this many candidates are offered for the model to choose
  from. Offered four on average, the model answered worse than when offered two.

### Changed

- Loading is decided by how sure verification is of its pick: `load_at` is compared against the pick and
  now defaults to 0.9. The per-candidate "fits" answers only admit candidates (`suggest_at`, now 0.2); a
  trace without a pick uses the fit of its candidate instead. On the bank testbed a pick this sure was
  right in 90% of requests, and wrong loads fell from 13% (0.1.0) to 6%; below it, the model gets a short
  list. A wrong skill loaded costs more than none: 71% of such turns were answered correctly, against 84%
  with nothing loaded.
- `Settings.skip_verify_at` defaults to 0.9: a ranking this sure of its first candidate is loaded without
  the second call, which spared it in 44% of requests with the same share of right loads.
- When verification fails, up to `max_suggest` ranked candidates are offered, instead of a fixed three.

## [0.2.0] - 2026-09-22

### Added

- `Settings.pick_question` and `Trace.picked`: verification now also asks which candidate is the right
  one, with the candidates' texts side by side, and records the answer.

### Changed

- Verification decides which skill to load by its pick among the candidates; the per-candidate "fits"
  answers only decide whether any is loaded (`load_at` against the best of them) and which are suggested
  (`suggest_at`). Ordered by "fits" alone, lookalike skills tied and the wrong one was often loaded. A
  trace recorded without a pick is ordered by the ranking.
- `Settings.max_load` defaults to 1: two lookalike skills loaded together gave worse answers than one.
- The loaded skill and the suggested ones now go in a message right after the user's request instead of
  the system message, which stays the same on every call. The provider's prompt cache keeps the
  conversation before the request; with the skills in the system message it was read again uncached on
  every turn. The message says the skill may be ignored if it does not fit, following TypeSafe's
  skill-suggestion cookbook.
- `recent_context`, the default context for the judge, now carries what the user asked and the agent
  answered, without tool calls and their results, which pushed the previous request out of the window.
- Settings are checked against the provider's limit with verification's pick counted in: it is now the
  longest question, carrying every candidate's description and the head of its text.

- Clarified README positioning: a deepagents extension built on LangChain middleware, with an
  independently usable routing core.
- Updated README benchmark results for the PyPI 0.1.0 run, including per-turn costs and sample size.
- PyPI publication is now started manually through Publish with an existing release tag. Release only
  creates the GitHub release, keeping publication attestations tied to the configured Trusted Publisher.

## [0.1.1] - 2026-09-21

### Fixed

- Instruction read errors now fall back to the ordinary full skills catalog for that model call,
  without injecting partially loaded instructions or retrying model errors.
- Skill selection, instruction loading and `find_skill` now use the current execution's catalog,
  including updated descriptions and paths, without sharing a cached router between concurrent runs.

### Changed

- Refreshed the README with a visual overview, a compact quick start and links to detailed documentation.

## [0.1.0] - 2026-09-20

The first release.

### Added

- `SkillRouter`, which decides on each turn which skills an agent should see: ranking over the catalog
  with "is a skill needed at all" asked alongside it, then verification of the candidates against their
  own text. Every threshold and every question is a `Settings` field. A turn is decided on its own, so
  the library needs no checkpointer and keeps nothing between turns.
- `LoadoutSkillsMiddleware` for deepagents agents, which takes the place of `SkillsMiddleware` and
  falls back to it whenever a decision fails.
- The `Judge` port, with `JevJudge` behind the optional `jev` extra, so the library depends on no
  provider SDK by default.
- `langchain_loadout.testing`: `check_judge` runs an adapter against the contract in one call, and
  `ScriptedJudge` answers from a script so an agent can be tested without a provider.
- An error taxonomy: `JudgeUnavailable` falls back to the full catalog, `JudgeMisconfigured` surfaces
  instead of hiding a broken setup, and both derive from `LoadoutError`.
- A public API: everything listed in `langchain_loadout.__all__`, importable from the package root and
  pulling in neither a framework nor a provider SDK.

[Unreleased]: https://github.com/deyna256/langchain-skill-router/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/deyna256/langchain-skill-router/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/deyna256/langchain-skill-router/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/deyna256/langchain-skill-router/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/deyna256/langchain-skill-router/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/deyna256/langchain-skill-router/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/deyna256/langchain-skill-router/releases/tag/v0.1.0
