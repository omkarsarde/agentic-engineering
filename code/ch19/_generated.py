# Auto-generated from chapters/19-protocols-frameworks.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable


class ProtocolError(RuntimeError):
    """A typed protocol, validation, or authorization failure.

    It carries a JSON-RPC error code so the caller can tell a contract mismatch
    (invalid params) apart from a domain result apart from a transport failure.
    Collapsing those three into one string called ``error`` forces the model to
    invent a retry policy it has no basis for.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Principal:
    """Who is calling, established by the transport rather than by arguments.

    A remote server derives this from a validated access token; a local stdio
    server inherits it from the launching host. The model never fills in
    ``tenant_id`` or ``scopes`` — identity a model can type is identity an
    attacker can forge through the model.
    """

    subject: str
    tenant_id: str
    audience: str
    scopes: frozenset[str]


@dataclass(frozen=True)
class Tool:
    """A discoverable operation and the single scope a caller must hold for it.

    ``description`` and ``schema`` are what the host shows the model to decide
    whether to propose the call; ``scope`` is what the server checks before
    running it. Keeping the two separate is the whole idea: what the model reads
    never becomes what the model is allowed to do.
    """

    name: str
    description: str
    schema: dict[str, Any]
    scope: str
    handler: Callable[[dict[str, Any], Principal], dict[str, Any]]


def _require_keys(arguments: dict[str, Any], required: set[str]) -> None:
    """Reject arguments that do not exactly match a schema's required keys."""
    missing, extra = required - arguments.keys(), arguments.keys() - required
    if missing or extra:
        raise ProtocolError(-32602, f"invalid arguments: missing={sorted(missing)} extra={sorted(extra)}")


class Server:
    """A minimal MCP-shaped server over JSON-RPC 2.0.

    The wire format is a small JSON-RPC subset — ``initialize``,
    ``tools/list``, ``resources/read``, ``tools/call`` — carried as text. The
    server negotiates one pinned protocol revision, filters discovery by the
    caller's scopes, reauthorizes every tool call even though the host already
    gated it, and stores receipts so a replayed write returns the same result.
    """

    PROTOCOL_VERSION = "2025-11-25"
    AUDIENCE = "support://mcp"

    def __init__(self) -> None:
        self.receipts: dict[str, dict[str, Any]] = {}
        self.tools: dict[str, Tool] = {
            "policy_lookup": Tool(
                "policy_lookup",
                "Read the current refund policy for the authenticated tenant.",
                {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
                "policy:read",
                self._policy_lookup,
            ),
            "case_tag": Tool(
                "case_tag",
                "Apply a reviewed routing tag to a support case.",
                {"type": "object",
                 "properties": {"case_id": {"type": "string"},
                                "tag": {"enum": ["manual_review", "eligible"]},
                                "idempotency_key": {"type": "string"}},
                 "required": ["case_id", "tag", "idempotency_key"]},
                "case:write",
                self._case_tag,
            ),
        }

    def _authorize(self, principal: Principal, scope: str) -> None:
        if principal.audience != self.AUDIENCE:
            raise ProtocolError(-32001, "invalid token audience")
        if scope not in principal.scopes:
            raise ProtocolError(-32002, f"missing scope: {scope}")

    def initialize(self, requested_version: str) -> dict[str, Any]:
        """Negotiate one explicit revision; reject anything but the pinned version."""
        if requested_version != self.PROTOCOL_VERSION:
            raise ProtocolError(-32600, f"unsupported protocol version: {requested_version!r}")
        return {"protocolVersion": self.PROTOCOL_VERSION,
                "serverInfo": {"name": "support-contracts", "version": "1.0.0"},
                "capabilities": {"tools": {}, "resources": {}}}

    def list_tools(self, principal: Principal) -> dict[str, Any]:
        """Return only the tools this principal is scoped to see."""
        self._authorize(principal, "policy:read")
        visible = [t for t in self.tools.values() if t.scope in principal.scopes]
        return {"tools": [{"name": t.name, "description": t.description, "inputSchema": t.schema}
                          for t in sorted(visible, key=lambda t: t.name)]}

    def read_resource(self, uri: str, principal: Principal) -> dict[str, Any]:
        """Return application-controlled context behind a stable URI."""
        self._authorize(principal, "policy:read")
        if uri != "policy://refund/current":
            raise ProtocolError(-32602, f"unknown resource: {uri}")
        return {"contents": [{"uri": uri, "mimeType": "application/json",
                              "text": '{"version":"v3","manual_review_above_cents":5000}'}]}

    def call_tool(self, name: str, arguments: dict[str, Any], principal: Principal) -> dict[str, Any]:
        """Reauthorize and execute one discovered operation."""
        tool = self.tools.get(name)
        if tool is None:
            raise ProtocolError(-32601, f"unknown tool: {name}")
        self._authorize(principal, tool.scope)
        _require_keys(arguments, set(tool.schema["required"]))
        result = tool.handler(arguments, principal)
        return {"content": [{"type": "text", "text": json.dumps(result)}], "structuredContent": result}

    def handle(self, message: str, principal: Principal) -> str:
        """Parse one JSON-RPC request line, dispatch it, and return one response line."""
        request = json.loads(message)
        try:
            if request.get("jsonrpc") != "2.0" or "id" not in request:
                raise ProtocolError(-32600, "request must be correlated JSON-RPC 2.0")
            method, params = request["method"], request.get("params", {})
            if method == "initialize":
                result = self.initialize(params["protocolVersion"])
            elif method == "tools/list":
                result = self.list_tools(principal)
            elif method == "resources/read":
                result = self.read_resource(params["uri"], principal)
            elif method == "tools/call":
                result = self.call_tool(params["name"], params["arguments"], principal)
            else:
                raise ProtocolError(-32601, f"unknown method: {method}")
            response: dict[str, Any] = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except ProtocolError as error:
            response = {"jsonrpc": "2.0", "id": request.get("id"),
                        "error": {"code": error.code, "message": error.message}}
        return json.dumps(response)

    def _policy_lookup(self, arguments: dict[str, Any], principal: Principal) -> dict[str, Any]:
        if arguments["topic"] != "refund":
            return {"found": False}
        return {"found": True, "policy_version": "v3", "manual_review_above_cents": 5000}

    def _case_tag(self, arguments: dict[str, Any], principal: Principal) -> dict[str, Any]:
        key = arguments["idempotency_key"]
        if key in self.receipts:
            return self.receipts[key]
        receipt = {"case_id": arguments["case_id"], "tag": arguments["tag"], "receipt": key}
        self.receipts[key] = receipt
        return receipt


@dataclass
class Client:
    """A host-owned connection to exactly one server.

    The host keeps one client per server, which is why a server cannot see the
    whole thread or discover its siblings. The client frames each call as
    JSON-RPC, correlates the response ``id`` with the request ``id``, and turns a
    server error object back into a raised ``ProtocolError`` so control flow —
    not string-matching — handles failures.
    """

    transport: Callable[[str], str]
    principal: Principal
    next_id: int = 0
    log: list[str] = field(default_factory=list)

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one correlated JSON-RPC request and return its result payload."""
        self.next_id += 1
        wire = json.dumps({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}})
        response = json.loads(self.transport(wire))
        if response.get("id") != self.next_id:
            raise ProtocolError(-32000, "response id did not match request id")
        if "error" in response:
            raise ProtocolError(response["error"]["code"], response["error"]["message"])
        self.log.append(method)
        return response["result"]

    def initialize(self) -> dict[str, Any]:
        return self.request("initialize", {"protocolVersion": Server.PROTOCOL_VERSION})

    def list_tools(self) -> list[str]:
        return [t["name"] for t in self.request("tools/list")["tools"]]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})["structuredContent"]


def connect(server: Server, principal: Principal) -> Client:
    """Bind a client to a server over an in-process transport carrying JSON text.

    The transport closes over ``principal``, modeling the rule that identity is
    asserted by the authenticated channel — a validated token, or a stdio parent
    process — and is never read from model-supplied arguments.

    Args:
        server: The server whose ``handle`` receives the framed messages.
        principal: The identity the transport asserts for every request.

    Returns:
        A ``Client`` bound to that server and identity.
    """
    return Client(lambda message: server.handle(message, principal), principal)


INJECTION_MARKERS = ("ignore previous", "ignore all", "disregard", "reveal", "exfiltrate", "send your")


def scan_for_injection(text: str) -> list[str]:
    """Flag imperative-override phrases in untrusted, server-supplied text.

    This is a backstop, not a boundary. A tool description is data, so the
    durable fix is to keep authority out of it entirely and to pin the tool's
    fingerprint; the scan only catches the obvious phrasings and buys review
    time against the rest.

    Args:
        text: A tool description or other server-supplied string.

    Returns:
        The marker phrases found in ``text``, in the order listed.
    """
    lowered = text.lower()
    return [marker for marker in INJECTION_MARKERS if marker in lowered]


def tool_fingerprint(tool: Tool) -> str:
    """Hash a tool's identity-defining fields so a later change is detectable.

    A rug-pull swaps a tool's behavior after approval by editing its description
    or schema. Hashing the name, description, and canonical schema turns that
    silent edit into a fingerprint mismatch the host catches on re-discovery, so
    an authority-expanding change forces re-review instead of sliding through.

    Args:
        tool: The tool to fingerprint.

    Returns:
        A hex SHA-256 digest over the tool's name, description, and schema.
    """
    material = json.dumps({"name": tool.name, "description": tool.description, "schema": tool.schema},
                          sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


TRUSTED_CATALOG: dict[str, set[str]] = {
    "PolicyCard": {"title", "body", "version"},
    "DataTable": {"columns", "rows"},
    "ApprovalForm": {"case_id", "action", "preview"},
}


def validate_component(name: str, props: dict[str, Any]) -> dict[str, Any]:
    """Validate a model- or server-proposed UI component against a trusted catalog.

    Generative UI must project data into known components, never execute
    model-authored code. A proposal passes only if the component name is in the
    catalog, its props are a subset of the allowed keys, and no prop value
    carries a script or ``javascript:`` payload; otherwise it is rejected to a
    safe fallback card the application owns.

    Args:
        name: The requested component name.
        props: The proposed properties for that component.

    Returns:
        ``{"ok": True, "component": ...}`` when safe, else ``{"ok": False,
        "reason": ..., "fallback": "ErrorCard"}``.
    """
    allowed = TRUSTED_CATALOG.get(name)
    if allowed is None:
        return {"ok": False, "reason": f"unknown component: {name}", "fallback": "ErrorCard"}
    if unexpected := (set(props) - allowed):
        return {"ok": False, "reason": f"unexpected props: {sorted(unexpected)}", "fallback": "ErrorCard"}
    risky = [k for k, v in props.items()
             if isinstance(v, str) and ("<script" in v.lower() or "javascript:" in v.lower())]
    if risky:
        return {"ok": False, "reason": f"script in props: {risky}", "fallback": "ErrorCard"}
    return {"ok": True, "component": {"name": name, "props": props}}


@dataclass
class CheckpointLog:
    """An append-only list of immutable state snapshots for one thread.

    Each snapshot is deep-copied on write, so a later mutation of the live state
    cannot reach back and rewrite history. That immutability is what makes
    ``fork`` (time travel) safe: branching from an old checkpoint leaves the
    original untouched. It exposes semantics, not durability — a real
    checkpointer adds concurrency control, encryption, and retention.
    """

    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def save(self, state: dict[str, Any]) -> None:
        """Append a deep copy of ``state`` as the next checkpoint."""
        self.snapshots.append(deepcopy(state))

    def fork(self, index: int, **updates: Any) -> dict[str, Any]:
        """Branch from checkpoint ``index`` with edited fields, leaving history intact."""
        state = deepcopy(self.snapshots[index])
        state.update(updates)
        return state


class GraphHost:
    """An explicit state graph over the same protocol client.

    A node reads one state snapshot and emits an update; a checkpoint is saved
    before the review node so the run pauses at a named interrupt rather than
    inside an opaque loop. ``resume`` takes a typed human decision and either
    applies the reviewed effect or records a rejection. The effect runs only
    after approval, so replaying the review node cannot write twice.
    """

    def __init__(self, client: Client, checkpoints: CheckpointLog | None = None) -> None:
        self.client = client
        self.checkpoints = checkpoints or CheckpointLog()

    def start(self, case_id: str) -> dict[str, Any]:
        """Run discover -> lookup -> review, checkpointing, and pause at the interrupt."""
        state: dict[str, Any] = {"node": "discover", "case_id": case_id, "status": "running"}
        self.client.initialize()
        state.update(node="lookup", tools=self.client.list_tools())
        self.checkpoints.save(state)
        policy = self.client.call_tool("policy_lookup", {"topic": "refund"})
        state.update(node="review", policy_version=policy["policy_version"], status="awaiting_approval")
        self.checkpoints.save(state)
        return deepcopy(state)

    def resume(self, state: dict[str, Any], approved: bool) -> dict[str, Any]:
        """Apply a reviewer decision at the interrupt; only approval writes an effect."""
        if state.get("status") != "awaiting_approval":
            raise ProtocolError(-32000, "state is not at the review interrupt")
        resumed = deepcopy(state)
        if not approved:
            resumed.update(node="finish", status="rejected", answer="reviewer rejected routing")
        else:
            receipt = self.client.call_tool(
                "case_tag", {"case_id": resumed["case_id"], "tag": "manual_review",
                             "idempotency_key": f"tag:{resumed['case_id']}:v3"})
            resumed.update(node="finish", status="completed", receipt=receipt,
                           answer=f"{resumed['case_id']} routed to {receipt['tag']} under policy v3")
        self.checkpoints.save(resumed)
        return resumed


@dataclass
class ReplayModel:
    """A deterministic stand-in for a real model, driven by a fixed script.

    A production loop sends the message history and tool schemas to a model and
    receives back either a tool call or a final answer. To stay offline and
    reproducible we replay a scripted list of those decisions; the script is
    visible, which is the honest way to teach the loop's control flow without a
    key, a network, or hidden nondeterminism.
    """

    script: list[dict[str, Any]]
    step: int = 0

    def decide(self, messages: list[dict[str, Any]], tools: list[str]) -> dict[str, Any]:
        """Return the next scripted decision (a real model would read the inputs)."""
        decision = self.script[self.step]
        self.step += 1
        return decision


class LoopHost:
    """A model-driven loop over the same protocol client the graph uses.

    The model proposes a tool call, the harness executes it against the server
    and appends the observation to the message history, and the loop repeats
    until the model returns a final answer. Control flow lives in the model
    inside harness gates — the third paradigm — but the server contract is
    unchanged, which is the whole point of a stable protocol seam.
    """

    def __init__(self, client: Client, model: ReplayModel) -> None:
        self.client = client
        self.model = model

    def run(self, case_id: str) -> dict[str, Any]:
        """Drive the loop to a final answer, returning it with the tool-call trace."""
        self.client.initialize()
        tools = self.client.list_tools()
        messages: list[dict[str, Any]] = [{"role": "user", "content": f"route case {case_id}"}]
        calls: list[str] = []
        while True:
            decision = self.model.decide(messages, tools)
            if decision["type"] == "final":
                return {"status": "completed", "answer": decision["answer"].format(case_id=case_id),
                        "calls": calls}
            name = decision["tool"]
            arguments = {k: (v.format(case_id=case_id) if isinstance(v, str) else v)
                         for k, v in decision["arguments"].items()}
            if name not in tools:
                raise ProtocolError(-32601, f"model proposed an undiscovered tool: {name}")
            observation = self.client.call_tool(name, arguments)
            calls.append(name)
            messages.append({"role": "tool", "name": name, "content": json.dumps(observation)})
