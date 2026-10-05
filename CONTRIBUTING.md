# Contributing to Skill Router

Use plain English in issues, commits, documentation and code comments. Follow the
[development guide](docs/development.md) for code and tests, [design](docs/design.md) for how the
product behaves, and the [Code of Conduct](CODE_OF_CONDUCT.md) when working with others.

## No signatures

Do not add generated-by lines, tool attribution footers or similar signatures anywhere: not in commits,
pull requests, reviews, comments, issues, documentation or code.

## Issues

Check for an existing issue before opening one. Give each issue one clear outcome and a short title
that describes the problem or intended change.

Report a security vulnerability privately, as the [security policy](SECURITY.md) describes, not in an issue.

Use the [issue template](.github/ISSUE_TEMPLATE/issue.md), which has these sections:

- **Problem:** explain what happens today and what is missing or wrong. For a bug, include reproduction
  steps, expected and actual behaviour, and the versions of Python, `langchain-skill-router` and the judge
  adapter in use.
- **Why it matters:** describe the effect on users or development. Explain why the work is useful
  without repeating the problem.
- **Done when:** list observable results that will make the task complete, including the tests or
  checks needed to verify them.

Keep the issue self-contained. Add examples, relevant links and scope limits when useful. Separate
agreed behaviour from proposals; do not invent implementation details to fill out the description.
Keep credentials, user requests and skill contents out of examples and logs.

Give each issue one type label: `bug`, `feature`, `maintenance`, `docs` or `question`. Add `ci` for CI
and developer tooling, and `good first issue` for small, self-contained tasks. Close duplicates and
declined issues with the matching close reason instead of a label.

## Branches

For issue work, branch from `main` and use only the issue number as the branch name. For issue #1:

```sh
git switch main
git switch -c 1
```

Use `1`, not `issue-1`, `feature/1` or a descriptive suffix. Keep the branch focused on its issue.

Work without an issue — repository setup, a typo, a dependency bump — uses a short descriptive branch
name instead, such as `changelog-typo`.

## Commits

For issue work, use one line in this format:

```text
#<issue number>: <change>
```

Examples:

```text
#1: add the judge conformance kit
#1: test that a missing limits property fails the check
```

Use a short English description that starts with an action, such as `add`, `fix` or `test`. Describe
the change, not the work session. Put longer explanations in the issue or pull request.

Work without an issue uses the same one line without the number:

```text
add the pull request template
```

Do not add a commit body, co-author lines, sign-off trailers or any other [signature](#no-signatures).
Git still records the normal author and committer metadata. Keep unrelated changes out of the commit and
inspect the staged diff before committing.

Write the [changelog](CHANGELOG.md) entry in the same commit as the change it describes.

## Checks

Use the commands in the [Justfile](Justfile):

| command | what it does |
|---|---|
| `just test` | runs the test suite |
| `just lint` | checks formatting, style and import order, changing nothing |
| `just format` | formats the code and applies the fixes ruff can make on its own |
| `just type` | checks types with the optional dependencies installed |
| `just build` | builds the wheel and source distribution |

Run `just format` first and `just lint` after it: formatting and the fixes ruff makes on its own are
applied by the first command, so whatever the second one reports needs a person. Both commands stop at
the first failing check and exit non-zero, and `just format` also exits non-zero when it has fixed what
it can and something is left over.

Run the relevant checks before submitting changes. Documentation-only changes need link and formatting
checks, not tests.

The [CI workflow](.github/workflows/ci.yml) runs `just lint` and `just type` once and `just test` on
every supported Python version, for pull requests to `main` and pushes to `main`. It sets `UV_LOCKED`,
so a stale `uv.lock` fails the build. `just format` changes files and is never run in CI.

Tests must not call a provider. The library has no live tests: `langchain_skill_router.testing` provides a
scripted judge for that, and the checks that do need a real provider — `check_judge` against an adapter,
and measurements against real models — belong to the bank testbed, which lives in its own repository.

## Releases

A release is cut by hand, from the Actions tab: run the **Release** workflow on `main`. It refuses to
run from another branch, and refuses a version that is already released, so the one thing to do first
is raise `version` in `pyproject.toml` and write the entry in the [changelog](CHANGELOG.md).

The workflow then runs the same checks CI runs, builds the wheel and the source distribution, attests
what it built, and publishes a GitHub release with those files attached. Publishing the release starts
the **Publish** workflow, which takes the files off the release — so what reaches the index is byte for
byte what the release holds — and uploads them to PyPI through trusted publishing. No token is stored
anywhere: GitHub's own identity for that run is exchanged for a short-lived credential, which is why
the job names the `pypi` environment that the publisher configuration on PyPI expects.

## Pull requests

Use the [pull request template](.github/pull_request_template.md) with these sections:

- **Problem:** explain what is missing or wrong and why the change is needed.
- **Changes:** describe the resulting behaviour and the decisions needed to review it.
- **Validation:** state which checks passed or could not run, and what behaviour the tests cover.

Keep each section short and avoid repeating the issue or listing every changed file. Add sections only
when needed, such as migration steps or breaking changes. Link the issue; use `Closes #<number>` when
the PR completes it. Update the description and affected documentation when the code changes. Do not
[sign](#no-signatures) the description.

Open the pull request as a draft while the work is in progress. When it is ready, mark it ready for review
and request a review from [@deyna256](https://github.com/deyna256) or another maintainer. Do this again after you address
the review's findings, so the reviewer knows the pull request is ready for another look.

## Reviews

Review the change as submitted, against the issue and the
[review checklist](docs/development.md#review-checklist). State facts: what the code does, what follows
from it, and what you ran. Do not restate the diff or guess at intent; ask instead.

Put every code-anchored finding in an inline comment on the line it concerns and keep the review body
for the verdict and for anything that spans files. Mark a finding that is not blocking with one of
these prefixes:

- **Nit:** small or stylistic; the author may skip it.
- **Question:** you need an answer before you can judge the code.

Everything else blocks the merge until it is resolved.

Use these sections in the review body, and omit a section with nothing in it:

- **Summary:** one sentence on what the change does and whether it is ready.
- **Blocking:** one item per problem, each with `file:line`, the consequence and a concrete fix or check.
- **Non-blocking:** nits, questions and follow-ups worth recording.
- **Checked:** the commands you ran and anything you could not verify.

Request changes only when a finding is blocking. Approve when the rest are nits and say which ones you
expect to be handled. Take work outside the scope of the pull request to a separate issue and link it
instead of growing the review. Do not [sign](#no-signatures) the review or its comments.

## Documentation

Keep shared documentation in Markdown. Distinguish planned behaviour from what the code implements.

| document | covers |
|---|---|
| [README](README.md) | what the library is, how to install and use it, what the measurements showed |
| [docs/design.md](docs/design.md) | how the decision is made and why it is made that way |
| [docs/development.md](docs/development.md) | coding rules for this repository |

Use short ADRs in `docs/decisions/` for significant decisions: status, context, decision, and
alternatives with consequences. Include sources when they explain the choice. This follows
[Nygard's ADR guidance](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

New decisions are **Proposed** until agreed. **Accepted** does not mean implemented. Update decisions to
match the agreed design. Remove obsolete records and fix their references; do not maintain an archive of
superseded designs. Git retains the history. Do not renumber surviving ADRs to fill gaps.

Update affected docs and links in the same change. Give each rule one home; other docs and LLM
instructions should link there rather than copy it.

### Local working documents

Use `.local/` at the repository root for research, drafts and development plans; for example,
`.local/research/` and `.local/plans/`. Git ignores this directory. Do not commit it or force-add its
contents.

Move accepted decisions and their essential reasons into shared documentation. Local proposals do not
become requirements without agreement. Shared docs must not require or link to local notes: a fresh
clone must contain everything needed to understand and work on the project.
