"""Skill Router for deepagents: a wrapper around the ordinary skills middleware (`SkillsMiddleware`).

Skills are still discovered by the ordinary middleware, through `state["skills_metadata"]`. On each new
user message Skill Router decides which of them are needed, and the model sees only those. On any failure the
request passes to the ordinary middleware untouched, so the model sees the full list exactly as it would
without Skill Router.

What the model sees is laid out for the provider's prompt cache, which matches a request from its start:
a new request is served from the cache only as far as it repeats an earlier one. The system message gets a
section that is the same on every call, and the turn's skills go in a message right after the user's
request, which is written into the conversation. Put in the system message, they changed the head of every
turn's first request (37% of that call's input came from the cache on the bank testbed, against 94% with the
full list). Added to the model request alone, they vanished on the next turn, so no earlier request was a
prefix of the new one: on the testbed the conversation came from the cache at the first call of a turn in
16-24% of turns, against 64% without the message. The message is marked, never taken for the user's request
and kept out of the context the judge reads. A skill loaded on an earlier turn is written again rather than
referred to: summarization may have replaced that turn in what the model sees while the state still holds it.

Selection, instruction reads and `find_skill` use the current execution's catalog. No catalog or router
is cached on the middleware instance, which may be shared by concurrent executions.

Wiring: `create_deep_agent(..., skills=[...], middleware=[SkillRouterMiddleware(...)])`. The wrapper
carries the built-in middleware's name and takes its place. The optimisation applies when the agent is
run asynchronously (`ainvoke`, `astream`); a synchronous run takes the ordinary path.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, NotRequired, cast

from deepagents.backends import BackendProtocol
from deepagents.middleware.skills import SkillMetadata, SkillsMiddleware
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse, OmitFromSchema
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool

from langchain_skill_router.core.judge import Judge
from langchain_skill_router.core.router import SkillRouter
from langchain_skill_router.core.types import DEFAULTS, Decision, Settings, Skill, Turn

logger = logging.getLogger(__name__)

PROMPT = """## Skills System

Skills are step-by-step instructions for recurring tasks. The catalog is large, so for each request the
skills picked for it are attached right after the request, with the instructions of the one that fits.
If none of them fits the task, call `find_skill` with a short description of what you need."""

# The first line is TypeSafe's skill-suggestion cookbook's: a suggestion that may be ignored, because
# pushing harder wins compliance on wrong suggestions too, and a wrong skill is worse than none.
LOADED = """Relevant to the current request: {name}. Ignore this if it does not fit what the user actually asked for.
Its instructions follow; there is no need to read its SKILL.md.

<skill name="{name}">
{text}
</skill>"""

LISTED = """Other skills that may fit (read one's SKILL.md with `read_file`, `limit=1000`, if you need it):

{skills_list}"""


PRIVATE = OmitFromSchema(input=True, output=True)  # internal to the middleware: not taken in, not returned


# The state of the middleware this one replaces, reached through its public `state_schema`: deepagents changes
# what `skills_metadata` accepts between releases (0.7.16 allows `None` on input to reload the catalog).
class SkillRouterState(SkillsMiddleware.state_schema):
    skill_router_turn: NotRequired[Annotated[str, PRIVATE]]  # id of the user message this decision was made for
    skill_router_failed: NotRequired[Annotated[bool, PRIVATE]]  # a failure means the ordinary full list


def is_skill_message(message: AnyMessage) -> bool:
    """The message Skill Router added after a request: not something the user said."""
    return "skill_router" in message.additional_kwargs


def skills_list(metadata: Sequence[SkillMetadata]) -> str:
    """Skills by name and description, each with the path to read, as the ordinary middleware lists them."""
    lines = []
    for m in metadata:
        lines.append(f"- **{m['name']}**: {m['description']}")
        if m.get("allowed_tools"):
            lines.append(f"  -> Allowed tools: {', '.join(m['allowed_tools'])}")
        lines.append(f"  -> Read `{m['path']}` for full instructions")
    return "\n".join(lines)


def with_section(system_message: SystemMessage | None, text: str) -> SystemMessage:
    """The system message with `text` added as its last block."""
    blocks = list(system_message.content_blocks) if system_message else []
    blocks.append({"type": "text", "text": f"\n\n{text}" if blocks else text})
    return SystemMessage(content_blocks=blocks)


def recent_context(messages: Sequence[AnyMessage], limit: int = 6) -> str:
    """The default context: the conversation leading up to the current user message — what the user asked
    and what the agent answered. Tool calls and their results are left out: after a turn with tools the last
    few messages are all tool output, and the previous request would drop out of the window."""
    said = [
        m
        for m in messages
        if (isinstance(m, HumanMessage) and not is_skill_message(m)) or (isinstance(m, AIMessage) and not m.tool_calls)
    ]
    lines = [f"{m.type}: {m.text[:500]}" for m in said if m.text.strip()]
    return "\n".join(lines[-limit:])


class SkillRouterMiddleware(SkillsMiddleware):
    state_schema = SkillRouterState

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        sources: Sequence[str],
        judge: Judge,
        settings: Settings = DEFAULTS,
        catalog_hint: str = "",
        context: Callable[[Sequence[AnyMessage]], str] = recent_context,
        on_decision: Callable[[Decision], None] | None = None,
    ) -> None:
        super().__init__(backend=backend, sources=sources)
        self.backend = backend  # skill texts are read through it
        self.judge, self.settings, self.context = judge, settings, context
        self.on_decision = on_decision  # every decision's trace, for logs, metrics and measurement
        self._decision_warning_logged = False
        self.tools: list[BaseTool] = [self._find_skill_tool(catalog_hint)]

    @property
    def name(self) -> str:
        return "SkillsMiddleware"  # takes the place of the built-in deepagents middleware

    # --- the decision: once per new user message ----------------------------------------------------

    async def abefore_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        # The base class fixes this parameter to its own state, while `state_schema = SkillRouterState` is what
        # the graph actually builds, so the narrowing has to be stated here rather than in the signature.
        ours = cast(SkillRouterState, state)
        messages = ours["messages"]
        last = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if isinstance(messages[i], HumanMessage) and not is_skill_message(messages[i])
            ),
            None,
        )
        if last is None:
            return None
        turn_id = messages[last].id or str(last)
        if ours.get("skill_router_turn") == turn_id:
            return None  # this turn already has a decision
        router = self._router_from(ours.get("skills_metadata") or [])
        turn = Turn(request=messages[last].text, context=self.context(messages[:last]))
        decision = await router.decide(turn)
        if self.on_decision:
            try:
                self.on_decision(decision)
            except Exception as err:
                # User metrics must not discard a decision. Their error text may contain private data.
                # One middleware can serve many turns; a broken callback must not flood the log.
                if not self._decision_warning_logged:
                    self._decision_warning_logged = True
                    logger.warning("Skill Router on_decision callback failed (%s); keeping the decision", type(err).__name__)
        if decision.trace.failure:
            return {"skill_router_turn": turn_id, "skill_router_failed": True}
        by_name = {m["name"]: m for m in (ours.get("skills_metadata") or [])}
        loaded = [n for n in decision.load if n in by_name]
        try:
            texts = {n: await self._text(by_name[n]["path"]) for n in loaded}
        except (OSError, UnicodeError) as err:
            logger.warning("Skill instructions could not be read; using the full catalog (%s)", type(err).__name__)
            return {"skill_router_turn": turn_id, "skill_router_failed": True}
        parts = [LOADED.format(name=n, text=t) for n, t in texts.items()]
        if listed := [by_name[n] for n in decision.suggest if n in by_name]:
            parts.append(LISTED.format(skills_list=skills_list(listed)))
        update: dict[str, Any] = {"skill_router_turn": turn_id, "skill_router_failed": False}
        if parts:
            # Right after the request: the turn's later calls and the next turns all start with it.
            update["messages"] = [HumanMessage("\n\n".join(parts), additional_kwargs={"skill_router": loaded})]
        return update

    # --- applying it: what the model sees -----------------------------------------------------------

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        state = request.state
        if "skill_router_turn" not in state or state.get("skill_router_failed"):
            return await super().awrap_model_call(request, handler)  # the ordinary path: the full list
        # The turn's skills are already in the conversation, after the request.
        return await handler(request.override(system_message=with_section(request.system_message, PROMPT)))

    # --- catalog and search -------------------------------------------------------------------------

    def _router_from(self, metadata: list[SkillMetadata]) -> SkillRouter:
        catalog = [Skill(m["name"], m["description"], self._reader(m["path"])) for m in metadata]
        return SkillRouter(catalog, self.judge, self.settings)

    def _reader(self, path: str) -> Callable[[], Awaitable[str]]:
        async def read() -> str:
            return await self._text(path)

        return read

    async def _text(self, path: str) -> str:
        """Read a skill's instructions from the backend. Deliberately not cached: an agent that runs for
        days would otherwise keep serving the text a SKILL.md had when it first read it."""
        [response] = await self.backend.adownload_files([path])
        if response.error or response.content is None:
            raise OSError(f"could not read {path}: {response.error}")
        return response.content.decode()

    def _find_skill_tool(self, catalog_hint: str) -> BaseTool:
        async def find_skill(query: str, runtime: ToolRuntime) -> str:
            metadata = runtime.state.get("skills_metadata")
            if metadata is None:
                return "Skill catalog is not loaded yet."
            if not metadata:
                return "No matching skill found."
            found = await self._router_from(metadata).search(query)
            if not found:
                return "No matching skill found."
            by_name = {m["name"]: m for m in metadata}
            return skills_list([by_name[s.name] for s in found])

        description = (f"{catalog_hint} " if catalog_hint else "") + (
            "Find skills (step-by-step instructions) for a task that the skills picked for the request don't cover. "
            "Pass a short description of the task; returns the best matching skills with paths to read."
        )
        return StructuredTool.from_function(coroutine=find_skill, name="find_skill", description=description)
