"""The Skill Router wrapper around the deepagents skills middleware: what the model sees, and what it sees on failure."""

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import FileDownloadResponse
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeCancelledError

from langchain_skill_router import Answer, Limits, Pick, Settings, YesNo
from langchain_skill_router.langchain import SkillRouterMiddleware, is_skill_message, recent_context
from langchain_skill_router.testing import ScriptedJudge, yes

SKILLS = {
    "visa-statement": "Statement for a visa: money movement, in English, stamped by the bank.",
    "spending-by-category": "The user's spending by category over a period.",
    "card-limits": "Card limits: withdrawals, transfers, purchases.",
}


class RecordingModel(GenericFakeChatModel):
    """A fake model: returns scripted replies and remembers what it was sent."""

    seen: list[list[Any]] = []  # noqa: RUF012 — a pydantic field, so each instance gets its own list

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def backend(tmp_path: Path) -> FilesystemBackend:
    for name, description in SKILLS.items():
        folder = tmp_path / "skills" / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            f'---\nname: {name}\ndescription: "{description}"\n---\n\n# {name}\nInstruction text for {name}.\n'
        )
    return FilesystemBackend(root_dir=tmp_path, virtual_mode=True)


def judge_choosing(name: str | None) -> ScriptedJudge:
    """Picks `name` with confidence, or fails when `name` is None."""

    def answer(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        if name is None:
            raise RuntimeError("provider unavailable")
        out: dict[str, Answer] = {}
        for key, q in questions.items():
            if isinstance(q, Pick):
                out[key] = Answer({o: (0.9 if o == name else 0.05) for o in q.options})
            else:
                out[key] = yes(0.95 if key in ("need", f"fits:{name}") else 0.05)
        return out

    return ScriptedJudge(answer)


async def run(backend: FilesystemBackend, judge: ScriptedJudge, replies: list[AIMessage]) -> RecordingModel:
    model = RecordingModel(messages=iter(replies))
    model.seen = []
    middleware = SkillRouterMiddleware(
        backend=backend, sources=["/skills/"], judge=judge, catalog_hint="Statements, spending, card limits."
    )
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware])
    await agent.ainvoke({"messages": [HumanMessage("I need a statement for the embassy")]})
    return model


def system_text(messages: list[Any]) -> str:
    return "\n".join(m.text for m in messages if isinstance(m, SystemMessage))


def prompt_text(messages: list[Any]) -> str:
    """Everything the model was sent, the system message and the conversation alike."""
    return "\n".join(m.text for m in messages)


async def test_confident_choice_shows_only_that_skill_and_its_text(backend):

    model = await run(backend, judge_choosing("visa-statement"), [AIMessage("done")])
    prompt = prompt_text(model.seen[0])

    assert "Instruction text for visa-statement" in prompt
    assert "card-limits" not in prompt and "spending-by-category" not in prompt


async def test_unsure_choice_lists_candidates_with_paths_to_read(backend, tmp_path):
    (tmp_path / "skills" / "visa-statement" / "SKILL.md").write_text(
        f'---\nname: visa-statement\ndescription: "{SKILLS["visa-statement"]}"\nallowed-tools: read_file grep\n---\n\n'
        "# visa-statement\nInstruction text for visa-statement.\n"
    )
    spread = {"visa-statement": 0.6, "spending-by-category": 0.3, "card-limits": 0.1}  # below `load_at`: listed, not loaded
    admitted = ("need", "fits:visa-statement", "fits:spending-by-category")

    def answer(state: Mapping[str, object], questions: Mapping[str, Pick | YesNo]) -> Mapping[str, Answer]:
        out: dict[str, Answer] = {}
        for key, q in questions.items():
            if isinstance(q, Pick):
                out[key] = Answer({o: spread[o] for o in q.options})
            else:
                out[key] = yes(0.95 if key in admitted else 0.05)
        return out

    model = await run(backend, ScriptedJudge(answer), [AIMessage("done")])
    prompt = prompt_text(model.seen[0])

    assert "Instruction text for" not in prompt
    assert (
        f"- **visa-statement**: {SKILLS['visa-statement']}\n"
        "  -> Allowed tools: read_file, grep\n"
        "  -> Read `/skills/visa-statement/SKILL.md` for full instructions"
    ) in prompt
    assert f"- **spending-by-category**: {SKILLS['spending-by-category']}\n  -> Read `" in prompt
    assert "card-limits" not in prompt


@pytest.mark.parametrize(
    "attachment",
    [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 5000}},
        {"type": "file", "file": {"filename": "statement.pdf", "file_data": "data:application/pdf;base64," + "B" * 5000}},
    ],
)
async def test_multimodal_request_sends_only_text_to_judge(backend, attachment):
    request = "I need a statement for the embassy"
    message = HumanMessage(content=[attachment, {"type": "text", "text": request}])
    judge = judge_choosing("visa-statement")
    model = RecordingModel(messages=iter([AIMessage("done")]))
    middleware = SkillRouterMiddleware(backend=backend, sources=["/skills/"], judge=judge)
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware])

    await agent.ainvoke({"messages": [message]})

    assert judge.calls
    assert all(state["request"] == request for state, _ in judge.calls)
    original = next(m for m in model.seen[0] if isinstance(m, HumanMessage) and not is_skill_message(m))
    assert original.text == request
    # LangChain normalizes file blocks before the model sees them.
    payload = "A" * 5000 if attachment["type"] == "image_url" else "B" * 5000
    assert payload in str(original.content)
    assert "Instruction text for visa-statement" in prompt_text(model.seen[0])


async def test_the_skill_follows_the_request_and_the_system_message_stays_the_same(backend):
    # The provider's cache matches a request from its start: whatever the turn picks must come after the
    # conversation so far, and the system message must not depend on the pick.
    visa = await run(backend, judge_choosing("visa-statement"), [AIMessage("done")])
    limits = await run(backend, judge_choosing("card-limits"), [AIMessage("done")])

    assert system_text(visa.seen[0]) == system_text(limits.seen[0])
    assert "Instruction text" not in system_text(visa.seen[0])
    request, skill = visa.seen[0][-2:]
    assert request.text == "I need a statement for the embassy"
    assert "Relevant to the current request: visa-statement" in skill.text
    assert "Instruction text for visa-statement" in skill.text


async def test_next_turn_starts_with_everything_the_previous_turn_sent(backend):
    # The skill message stays in the conversation: the provider's cache serves a request only as far as it
    # repeats an earlier one, so the next turn must begin with everything the previous one sent.
    model = RecordingModel(messages=iter([AIMessage("done"), AIMessage("done again")]))
    judge = judge_choosing("visa-statement")
    middleware = SkillRouterMiddleware(backend=backend, sources=["/skills/"], judge=judge)
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware], checkpointer=InMemorySaver())
    thread = {"configurable": {"thread_id": "t"}}

    await agent.ainvoke({"messages": [HumanMessage("I need a statement for the embassy")]}, thread)
    result = await agent.ainvoke({"messages": [HumanMessage("And one for the other embassy")]}, thread)

    first, second = model.seen
    assert [(m.type, m.text) for m in second[: len(first)]] == [(m.type, m.text) for m in first]
    assert "Instruction text for visa-statement" in second[-1].text  # written again, not referred to
    assert [judge_state["request"] for judge_state, _ in judge.calls][-1] == "And one for the other embassy"
    assert "Instruction text" not in "".join(str(s.get("context", "")) for s, _ in judge.calls)
    assert [m.text for m in result["messages"] if not is_skill_message(m)] == [
        "I need a statement for the embassy",
        "done",
        "And one for the other embassy",
        "done again",
    ]


async def test_failure_falls_back_to_the_usual_full_list(backend):

    model = await run(backend, judge_choosing(None), [AIMessage("done")])
    prompt = system_text(model.seen[0])

    assert all(name in prompt for name in SKILLS)
    assert "Instruction text for" not in prompt


@pytest.mark.parametrize("failure", ["missing", "empty_response", "encoding", "io", "timeout"])
async def test_instruction_read_failure_restores_full_catalog(backend, monkeypatch, caplog, failure):
    judge = ScriptedJudge(
        lambda state, questions: {
            key: Answer(dict.fromkeys(q.options, 1 / len(q.options))) if isinstance(q, Pick) else yes(0.95)
            for key, q in questions.items()
        }
    )
    original_download = backend.adownload_files

    def fail_after_selection(decision):
        assert len(decision.load) == 2 and decision.trace.failure is None
        broken_path = f"/skills/{decision.load[-1]}/SKILL.md"

        async def download(paths):
            if paths != [broken_path]:
                return await original_download(paths)
            if failure == "io":
                raise OSError("backend unavailable")
            if failure == "timeout":
                raise TimeoutError("backend timed out")
            return [
                FileDownloadResponse(
                    path=broken_path,
                    content=b"\xff" if failure == "encoding" else None,
                    error="file_not_found" if failure == "missing" else None,
                )
            ]

        monkeypatch.setattr(backend, "adownload_files", download)

    model = RecordingModel(
        messages=iter(
            [
                AIMessage("", tool_calls=[{"name": "ls", "args": {"path": "/"}, "id": "call-1"}]),
                AIMessage("done"),
            ]
        )
    )
    middleware = SkillRouterMiddleware(
        backend=backend,
        sources=["/skills/"],
        judge=judge,
        settings=Settings(max_load=2, load_at=0.3),
        on_decision=fail_after_selection,
    )
    agent = create_deep_agent(
        model=model, backend=backend, skills=["/skills/"], middleware=[middleware], system_prompt="Keep the original instructions."
    )
    await agent.ainvoke({"messages": [HumanMessage("I need a statement")]})

    assert len(model.seen) == 2
    for messages in model.seen:
        prompt = system_text(messages)
        assert all(name in prompt for name in SKILLS)
        assert "Keep the original instructions." in prompt
        assert "Instruction text for" not in prompt_text(messages)  # discard even the first, successfully read skill
        assert "Relevant to the current request" not in prompt_text(messages)
    assert "using the full catalog" in caplog.text


async def test_instruction_read_does_not_hide_backend_bugs(backend, monkeypatch):
    async def download(paths):
        raise TypeError("backend bug")

    middleware = SkillRouterMiddleware(
        backend=backend,
        sources=["/skills/"],
        judge=judge_choosing("visa-statement"),
        on_decision=lambda decision: monkeypatch.setattr(backend, "adownload_files", download),
    )
    model = RecordingModel(messages=iter([AIMessage("done")]))
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware])
    with pytest.raises(TypeError, match="backend bug"):
        await agent.ainvoke({"messages": [HumanMessage("I need a statement")]})
    assert not model.seen


async def test_cancellation_during_instruction_read_stops_the_agent(backend, monkeypatch):
    reading = asyncio.Event()

    async def download(paths):
        reading.set()
        await asyncio.Future()

    middleware = SkillRouterMiddleware(
        backend=backend,
        sources=["/skills/"],
        judge=judge_choosing("visa-statement"),
        on_decision=lambda decision: monkeypatch.setattr(backend, "adownload_files", download),
    )
    model = RecordingModel(messages=iter([AIMessage("done")]))
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware])
    task = asyncio.create_task(agent.ainvoke({"messages": [HumanMessage("I need a statement")]}))
    try:
        async with asyncio.timeout(5):
            await reading.wait()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not model.seen


async def test_model_io_error_is_not_retried_as_a_skill_fallback(backend):
    class FailingModel(RecordingModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.seen.append(messages)
            raise OSError("model unavailable")

    model = FailingModel(messages=iter([]))
    middleware = SkillRouterMiddleware(backend=backend, sources=["/skills/"], judge=judge_choosing("visa-statement"))
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware])
    with pytest.raises(OSError, match="model unavailable"):
        await agent.ainvoke({"messages": [HumanMessage("I need a statement")]})
    assert len(model.seen) == 1
    assert "Instruction text for visa-statement" in prompt_text(model.seen[0])


async def test_decision_is_made_once_per_turn(backend):
    judge = judge_choosing("visa-statement")
    ls = AIMessage("", tool_calls=[{"name": "ls", "args": {"path": "/"}, "id": "call-1"}])

    model = await run(backend, judge, [ls, AIMessage("done")])

    assert len(model.seen) == 2  # the model was called twice in the turn
    assert len(judge.calls) == 2  # the judge once per turn: cheap call and ranking, sure enough to skip verification
    assert "Instruction text for visa-statement" in prompt_text(model.seen[1])


async def test_every_decision_is_reported_for_observability(backend):
    seen = []
    model = RecordingModel(messages=iter([AIMessage("done")]))
    middleware = SkillRouterMiddleware(
        backend=backend, sources=["/skills/"], judge=judge_choosing("visa-statement"), on_decision=seen.append
    )
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware])

    await agent.ainvoke({"messages": [HumanMessage("I need a statement for the embassy")]})

    assert [d.load for d in seen] == [("visa-statement",)]


@pytest.mark.parametrize("kind", [KeyError, RuntimeError])
async def test_decision_callback_failure_preserves_decision_and_redacts_warning(backend, caplog, kind):
    def on_decision(decision):
        raise kind("private metrics label")

    model = RecordingModel(messages=iter([AIMessage("done")]))
    middleware = SkillRouterMiddleware(
        backend=backend,
        sources=["/skills/"],
        judge=judge_choosing("visa-statement"),
        on_decision=on_decision,
    )
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware])

    await agent.ainvoke({"messages": [HumanMessage("I need a statement for the embassy")]})

    assert len(model.seen) == 1
    assert "Instruction text for visa-statement" in prompt_text(model.seen[0])
    assert len(caplog.records) == 1
    assert caplog.records[0].name == "langchain_skill_router.langchain.middleware"
    assert caplog.records[0].levelname == "WARNING"
    assert "on_decision" in caplog.text
    assert kind.__name__ in caplog.text
    assert "private metrics label" not in caplog.text
    assert caplog.records[0].exc_info is None


async def test_decision_callback_warns_once_per_middleware_but_runs_every_turn(backend, caplog):
    seen = []

    def on_decision(decision):
        seen.append(decision)
        raise RuntimeError("private metrics label")

    middleware = SkillRouterMiddleware(
        backend=backend,
        sources=["/skills/"],
        judge=judge_choosing("visa-statement"),
        on_decision=on_decision,
    )
    for _ in range(2):
        model = RecordingModel(messages=iter([AIMessage("done")]))
        agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware])
        await agent.ainvoke({"messages": [HumanMessage("I need a statement for the embassy")]})

    assert len(seen) == 2
    assert len(caplog.records) == 1


async def test_decision_callback_cancellation_propagates(backend):
    def on_decision(decision):
        raise asyncio.CancelledError

    middleware = SkillRouterMiddleware(
        backend=backend,
        sources=["/skills/"],
        judge=judge_choosing("visa-statement"),
        on_decision=on_decision,
    )
    model = RecordingModel(messages=iter([AIMessage("done")]))
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware])

    with pytest.raises(NodeCancelledError) as error:
        await agent.ainvoke({"messages": [HumanMessage("I need a statement for the embassy")]})
    assert isinstance(error.value.__cause__, asyncio.CancelledError)
    assert not model.seen


class CatalogMiddleware(AgentMiddleware):
    """Supply the invocation's catalog after ordinary discovery."""

    def __init__(self, metadata):
        self.metadata = metadata

    async def abefore_agent(self, state, runtime):
        return {"skills_metadata": self.metadata}


def catalog_agent(backend, middleware, tmp_path, label, name):
    path = f"/{label}/SKILL.md"
    folder = tmp_path / label
    folder.mkdir()
    (folder / "SKILL.md").write_text(f"Instructions for {label}")
    metadata = [{"name": name, "description": f"Description for {label}", "path": path, "allowed_tools": []}]
    model = RecordingModel(
        messages=iter(
            [
                AIMessage("", tool_calls=[{"name": "find_skill", "args": {"query": label}, "id": f"search-{label}"}]),
                AIMessage("done"),
            ]
        )
    )
    agent = create_deep_agent(model=model, backend=backend, skills=["/skills/"], middleware=[middleware, CatalogMiddleware(metadata)])
    return agent, model


async def test_catalog_updates_with_unchanged_skill_names(backend, tmp_path):
    judge = judge_choosing("statement")
    middleware = SkillRouterMiddleware(backend=backend, sources=["/skills/"], judge=judge)
    for label in ("old", "new"):
        agent, model = catalog_agent(backend, middleware, tmp_path, label, "statement")
        result = await agent.ainvoke({"messages": [HumanMessage(label)]})
        assert f"Instructions for {label}" in prompt_text(model.seen[0])
        search = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        assert f"Description for {label}" in search.content
        assert f"/{label}/SKILL.md" in search.content
    ranking_questions = [q for _, questions in judge.calls for q in questions.values() if isinstance(q, Pick)]
    assert ranking_questions[-1].options == {"statement": "Description for new"}


@pytest.mark.parametrize("same_name", [False, True])
async def test_overlapping_runs_keep_their_own_catalog(backend, tmp_path, same_name):
    a_started, b_finished = asyncio.Event(), asyncio.Event()

    async def answer(state, questions):
        if state["request"] == "A" and "need" in questions:
            a_started.set()
            await b_finished.wait()
        return {key: Answer(dict.fromkeys(q.options, 1.0)) if isinstance(q, Pick) else yes(0.95) for key, q in questions.items()}

    class OverlappingJudge:
        limits = Limits()

        async def ask(self, state, questions):
            return await answer(state, questions)

    middleware = SkillRouterMiddleware(backend=backend, sources=["/skills/"], judge=OverlappingJudge())
    agents = {label: catalog_agent(backend, middleware, tmp_path, label, "statement" if same_name else label) for label in ("A", "B")}

    async def invoke(label):
        if label == "B":
            await a_started.wait()
        try:
            return await agents[label][0].ainvoke({"messages": [HumanMessage(label)]})
        finally:
            if label == "B":
                b_finished.set()

    async with asyncio.timeout(5):
        results = await asyncio.gather(invoke("A"), invoke("B"))
    for label, result in zip(("A", "B"), results, strict=True):
        prompt = prompt_text(agents[label][1].seen[0])
        assert f"Instructions for {label}" in prompt
        assert f"Instructions for {'B' if label == 'A' else 'A'}" not in prompt
        search = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        assert f"Description for {label}" in search.content
        assert f"/{label}/SKILL.md" in search.content


def test_context_is_the_conversation_without_tool_traffic():
    messages = [
        HumanMessage("How much did I spend in June?"),
        AIMessage("", tool_calls=[{"name": "find_operations", "args": {}, "id": "call-1"}]),
        ToolMessage('{"operations": []}' * 50, tool_call_id="call-1"),
        AIMessage("You spent 41,000 roubles in June."),
    ]

    context = recent_context(messages)

    assert context == "human: How much did I spend in June?\nai: You spent 41,000 roubles in June."
