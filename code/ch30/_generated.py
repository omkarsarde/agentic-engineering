# Auto-generated from chapters/30-multimodal-voice-video-media.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import numpy as np


def kmeans(points: np.ndarray, k: int, seed: int, iters: int = 25) -> np.ndarray:
    """Fit ``k`` cluster centers with seeded Lloyd's algorithm.

    A codebook is exactly a set of cluster centers: k-means places ``k``
    representative points so that every sample is close to one of them, which
    is what a vector quantizer needs. Iteration is fixed and the seed is
    explicit so the codebook is reproducible.

    Args:
        points: Array of shape ``(n, d)`` to quantize.
        k: Number of codebook entries to learn.
        seed: Seed for the initial center draw.
        iters: Lloyd's-algorithm passes.

    Returns:
        Array of shape ``(k, d)`` holding the learned code vectors.
    """
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), size=k, replace=False)].copy()
    for _ in range(iters):
        assign = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (assign == j).any():
                centers[j] = points[assign == j].mean(0)
    return centers


def rvq_fit(points: np.ndarray, n_codebooks: int, k: int, seed: int):
    """Fit a residual-codebook stack and report error after each stage.

    Each stage quantizes the residual the earlier stages left behind, so the
    reconstruction refines the *same* frame instead of spelling more output.
    The returned errors are root-mean-square reconstruction errors and must
    fall monotonically: a later codebook can always encode a zero shift.

    Args:
        points: Latent vectors of shape ``(n, d)``.
        n_codebooks: Number of residual stages to stack.
        k: Entries per codebook.
        seed: Base seed; stage ``m`` uses ``seed + m``.

    Returns:
        A tuple ``(codebooks, errors, residuals)`` where ``errors[m]`` is the
        RMS error after ``m + 1`` stages and ``residuals[m]`` is the residual
        cloud at that stage.
    """
    residual = points.copy()
    recon = np.zeros_like(points)
    codebooks, errors, residuals = [], [], []
    for m in range(n_codebooks):
        cb = kmeans(residual, k, seed + m)
        assign = ((residual[:, None, :] - cb[None, :, :]) ** 2).sum(-1).argmin(1)
        recon = recon + cb[assign]
        residual = points - recon
        codebooks.append(cb)
        errors.append(float(np.sqrt((residual ** 2).sum(1).mean())))
        residuals.append(residual.copy())
    return codebooks, errors, residuals


def synth_latents(n: int, seed: int) -> np.ndarray:
    """Return ``n`` synthetic 2-D audio-like latents drawn from four clusters."""
    rng = np.random.default_rng(seed)
    centers = np.array([[1.2, 0.8], [-1.0, 1.1], [0.3, -1.3], [-0.9, -0.7]])
    return np.vstack([rng.normal(c, 0.35, size=(n // 4, 2)) for c in centers])


import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """Per-stage millisecond costs for one scripted turn.

    Fields are independent draws standing in for measured stage timers on a
    real call; ``s2s_ms`` is the single-model latency of a native
    speech-to-speech path that replaces ASR, the model, and TTS.
    """

    vad_ms: int
    asr_ms: int
    ttft_ms: int
    tts_ms: int
    net_ms: int
    s2s_ms: int


def latency_corpus(n: int, seed: int) -> list[Turn]:
    """Return ``n`` reproducible turns with plausible per-stage latencies.

    A seeded corpus stands in for a batch of measured calls: deterministic, so
    the median-passes-tail-misses story is the same on every run.

    Args:
        n: Number of turns to generate.
        seed: Seed for the per-stage latency draws.

    Returns:
        A list of ``Turn`` records.
    """
    rng = random.Random(seed)
    return [
        Turn(
            vad_ms=rng.randint(180, 340),
            asr_ms=rng.randint(120, 260),
            ttft_ms=rng.randint(260, 520),
            tts_ms=rng.randint(90, 180),
            net_ms=rng.randint(40, 110),
            s2s_ms=rng.randint(300, 560),
        )
        for _ in range(n)
    ]


def percentile(values, q: float) -> int:
    """Return the nearest-rank ``q``-quantile of an integer iterable.

    Nearest-rank keeps the result an observed sample, which is the honest
    thing to report for a small fixture: no interpolation invents a number
    between two measured turns.

    Args:
        values: Non-empty iterable of latencies.
        q: Quantile in ``[0, 1]``; ``0.95`` is the tail we care about.

    Returns:
        The latency at rank ``ceil(q * n)``.
    """
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def first_audio_paths(rows: list[Turn]) -> dict[str, list[int]]:
    """Compute first-audio latency for three architectures per turn.

    ``cascade_seq`` waits on every stage; ``cascade_overlap`` credits streaming
    ASR (only a 25% finalization tail survives endpoint) and sentence-level
    TTS (already folded into time-to-first-token); ``native_s2s`` collapses
    recognition, reasoning, and synthesis into one model latency.

    Args:
        rows: The turn corpus.

    Returns:
        A dict mapping each architecture to its per-turn first-audio latencies.
    """
    return {
        "cascade_seq": [t.vad_ms + t.asr_ms + t.ttft_ms + t.tts_ms + t.net_ms for t in rows],
        "cascade_overlap": [
            t.vad_ms + max(20, round(t.asr_ms * 0.25)) + t.ttft_ms + t.tts_ms + t.net_ms
            for t in rows
        ],
        "native_s2s": [t.vad_ms + t.s2s_ms + t.net_ms for t in rows],
    }


from enum import Enum


class Phase(str, Enum):
    """The two durable phases of a voice session."""

    LISTENING = "listening"
    RESPONDING = "responding"


@dataclass(frozen=True)
class AudioFrame:
    """One timestamped input-energy observation, in milliseconds."""

    at_ms: int
    energy: float


@dataclass(frozen=True)
class AudioChunk:
    """One output chunk labeled with the response that owns it."""

    response_id: int
    sequence: int
    text: str


@dataclass(frozen=True)
class ToolProposal:
    """A typed effect proposal reconstructed from speech, owned by a response."""

    response_id: int
    tool: str
    arguments: dict


class VoiceSession:
    """Own turn state, an interruptible output queue, and effect authority.

    The session is the durable part of a voice agent that a swappable ASR,
    model, or TTS adapter cannot be trusted to provide: it decides which audio
    is still eligible to play and which proposal may still cause an effect.

    Args:
        silence_ms: Quiet duration that closes an input turn.
        speech_energy: Energy at or above this counts as speech.

    Raises:
        ValueError: If either threshold is non-positive.
    """

    def __init__(self, silence_ms: int = 320, speech_energy: float = 0.35) -> None:
        if silence_ms <= 0 or speech_energy <= 0:
            raise ValueError("thresholds must be positive")
        self.silence_ms = silence_ms
        self.speech_energy = speech_energy
        self.phase = Phase.LISTENING
        self.active_response_id: int | None = None
        self.cancelled_response_ids: set[int] = set()
        self.output: list[AudioChunk] = []
        self.last_speech_ms: int | None = None
        self.effects: list[dict] = []
        self.events: list[str] = []

    def observe(self, frame: AudioFrame) -> bool:
        """Consume one input frame; return whether a user turn just closed.

        Speech refreshes the last-speech timestamp, and speech heard while the
        session is responding is a barge-in that cancels the current response
        before anything else. A turn closes only once silence since the last
        speech reaches ``silence_ms`` — the single knob the sweep varies.

        Args:
            frame: The next timestamped energy observation.

        Returns:
            ``True`` on the frame that closes a turn, else ``False``.
        """
        if frame.energy >= self.speech_energy:
            if self.phase is Phase.RESPONDING:
                self.cancel_for_barge_in()
            self.last_speech_ms = frame.at_ms
            return False
        if self.last_speech_ms is not None and frame.at_ms - self.last_speech_ms >= self.silence_ms:
            self.last_speech_ms = None
            self.events.append(f"turn:{frame.at_ms}:closed")
            return True
        return False

    def start_response(self) -> int:
        """Allocate the next response identity and begin accepting its chunks."""
        self.active_response_id = 1 if self.active_response_id is None else self.active_response_id + 1
        self.phase = Phase.RESPONDING
        self.events.append(f"response:{self.active_response_id}:started")
        return self.active_response_id

    def accept_chunk(self, chunk: AudioChunk) -> bool:
        """Queue a chunk only while its response still owns playback.

        Identity is checked before the cancelled-set because a chunk from a
        superseded response is rejected for the same reason whether or not it
        was explicitly cancelled: it no longer owns the floor.

        Args:
            chunk: An output chunk tagged with its response id.

        Returns:
            ``True`` if the chunk was queued, ``False`` if it was rejected.
        """
        if chunk.response_id != self.active_response_id:
            self.events.append(f"response:{chunk.response_id}:late_chunk_rejected")
            return False
        if chunk.response_id in self.cancelled_response_ids:
            self.events.append(f"response:{chunk.response_id}:cancelled_chunk_rejected")
            return False
        self.output.append(chunk)
        return True

    def cancel_for_barge_in(self) -> None:
        """Cancel the current response and clear only its queued audio."""
        rid = self.active_response_id
        if rid is None:
            return
        self.cancelled_response_ids.add(rid)
        self.output = [c for c in self.output if c.response_id != rid]
        self.events.append(f"response:{rid}:cancelled_and_cleared")
        self.phase = Phase.LISTENING

    def propose_refund(self, order_id: str, amount: int) -> ToolProposal:
        """Build a typed refund proposal without causing any effect."""
        if self.active_response_id is None:
            raise RuntimeError("a response must own the proposal")
        return ToolProposal(self.active_response_id, "refund_order",
                            {"order_id": order_id, "amount": amount})

    def confirm_and_execute(self, proposal: ToolProposal, confirmation: dict) -> str:
        """Commit an effect only if identity and exact arguments still match.

        The four return strings are the caller's decision surface:
        ``deny:superseded_response`` and ``deny:cancelled_response`` mean the
        proposal lost the floor; ``review:confirmation_mismatch`` means the
        spoken-back arguments differ from what would execute (the fifty-vs-
        fifteen case) and must not commit; ``allow`` is the only path that
        appends a receipt, exactly once.

        Args:
            proposal: The typed proposal recovered from speech.
            confirmation: The normalized arguments the user actually confirmed.

        Returns:
            One of the four decision strings above.
        """
        if proposal.response_id != self.active_response_id:
            return "deny:superseded_response"
        if proposal.response_id in self.cancelled_response_ids:
            return "deny:cancelled_response"
        if confirmation != proposal.arguments:
            return "review:confirmation_mismatch"
        self.effects.append({"tool": proposal.tool, **proposal.arguments})
        self.events.append(f"effect:{proposal.tool}:committed")
        return "allow"


def scripted_turn(mid_pause_ms: int, hop_ms: int = 20):
    """Return ``(frames, true_end_ms)`` for one turn with a mid-sentence pause.

    The turn speaks, pauses to think for ``mid_pause_ms``, speaks again, then
    goes silent. ``true_end_ms`` marks the real end of speech, so a close
    before it is a false cut and a close after it measures endpointing delay.

    Args:
        mid_pause_ms: Length of the thinking pause between the two speech runs.
        hop_ms: Frame period; frames are emitted every ``hop_ms`` milliseconds.

    Returns:
        The frame list and the millisecond timestamp of the true end of speech.
    """
    frames, t = [], 0
    def block(duration, energy):
        nonlocal t
        for _ in range(duration // hop_ms):
            frames.append(AudioFrame(t, energy))
            t += hop_ms
    block(600, 0.9)
    block(mid_pause_ms, 0.0)
    block(500, 0.9)
    true_end = t
    block(1200, 0.0)
    return frames, true_end


def endpoint_sweep(pauses: list[int], thresholds: list[int]) -> list[dict]:
    """Run the turn detector over scripted turns at several silence thresholds.

    For each threshold, a fresh session consumes each turn's frames until it
    first closes a turn. Closing before ``true_end`` is a false cut; closing
    after it contributes to mean endpointing latency. The two columns are the
    tradeoff a product must choose a point on.

    Args:
        pauses: Mid-sentence pause lengths to script, in milliseconds.
        thresholds: Silence thresholds to sweep.

    Returns:
        One row per threshold with ``false_cut_rate`` and ``endpoint_lat_ms``.
    """
    report = []
    for thr in thresholds:
        false_cuts, latencies = 0, []
        for pause in pauses:
            frames, true_end = scripted_turn(pause)
            session = VoiceSession(silence_ms=thr)
            closed_at = None
            for frame in frames:
                if session.observe(frame):
                    closed_at = frame.at_ms
                    break
            if closed_at is not None and closed_at < true_end:
                false_cuts += 1
            elif closed_at is not None:
                latencies.append(closed_at - true_end)
        report.append({
            "threshold_ms": thr,
            "false_cut_rate": round(false_cuts / len(pauses), 2),
            "endpoint_lat_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        })
    return report


def frame_budget(minutes: int, fps: int, tokens_per_frame: int, ctx_limit: int):
    """Return ``(frames, tokens, overflow_ratio)`` for dense video ingestion.

    Sending every frame at full resolution is the strawman the sampling
    strategies exist to avoid; the overflow ratio is how many times over a
    context window a naive pass would run.

    Args:
        minutes: Clip length in minutes.
        fps: Frames per second decoded.
        tokens_per_frame: Vision tokens one frame costs after tiling.
        ctx_limit: The model's context budget in tokens.

    Returns:
        The frame count, total token count, and the overflow ratio.
    """
    frames = minutes * 60 * fps
    tokens = frames * tokens_per_frame
    return frames, tokens, tokens / ctx_limit


frames, tokens, ratio = frame_budget(minutes=60, fps=30, tokens_per_frame=256, ctx_limit=1_000_000)
print(f"{frames:,} frames -> {tokens:,} tokens = {ratio:.0f}x a 1M-token context")


def uniform_sample(duration_s: int, k: int) -> list[int]:
    """Return ``k`` evenly spaced sample times over ``duration_s`` seconds."""
    return [round(i * duration_s / (k - 1)) for i in range(k)]


def shot_boundary_sample(duration_s: int, cuts: list[int], k: int) -> list[int]:
    """Sample at scene cuts first, then fill the budget with periodic coverage.

    Scene cuts catch visual change; periodic fill keeps a long static shot from
    disappearing. Neither is conditioned on the question, which is why a short
    event survives only by luck.

    Args:
        duration_s: Clip length in seconds.
        cuts: Scene-boundary timestamps.
        k: Total frame budget.

    Returns:
        Sorted, de-duplicated sample times.
    """
    picks = sorted(set(cuts))[:k]
    if len(picks) < k:
        picks += uniform_sample(duration_s, k - len(picks))
    return sorted(set(picks))[:k]


def agentic_seek(duration_s: int, event: tuple, k: int) -> list[int]:
    """Spend half the budget on a coarse scan, then seek densely near a hit.

    The coarse pass stands in for cheap evidence (thumbnails, ASR, OCR); a
    frame inside ``event`` is the cheap detector firing. Dense frames are then
    concentrated around that hit, which is what lets a small budget localize a
    short event a uniform grid steps over.

    Args:
        duration_s: Clip length in seconds.
        event: Ground-truth ``(start, end)`` the cheap detector can hit.
        k: Total frame budget.

    Returns:
        Sorted, de-duplicated sample times.
    """
    coarse = uniform_sample(duration_s, k // 2)
    hits = [t for t in coarse if event[0] <= t <= event[1]]
    used = list(coarse)
    if hits:
        c = hits[0]
        used += [c - 8 + 2 * i for i in range(8) if 0 <= c - 8 + 2 * i <= duration_s]
    else:
        near = min(coarse, key=lambda t: abs(t - (event[0] + event[1]) / 2))
        used += [near - 20 + 5 * i for i in range(9) if 0 <= near - 20 + 5 * i <= duration_s]
    return sorted(set(used))


def moment_recall(samples: list[int], event: tuple) -> int:
    """Return 1 if any sampled frame falls inside the event, else 0."""
    return int(any(event[0] <= t <= event[1] for t in samples))


def predicted_interval(samples: list[int], event: tuple):
    """Return the span of sampled frames that hit the event, or ``None``."""
    hits = [t for t in samples if event[0] <= t <= event[1]]
    return (min(hits), max(hits)) if hits else None


def t_iou(pred, gt: tuple) -> float:
    """Return temporal intersection-over-union of a predicted and true interval.

    Recall answers *did we find the moment*; tIoU answers *how tightly*. A
    single-frame hit recalls the event but has near-zero tIoU, which is why
    both metrics are reported: one guards against misses, the other against
    imprecise, unreviewable answers.

    Args:
        pred: The predicted ``(start, end)`` interval, or ``None`` if nothing hit.
        gt: The ground-truth ``(start, end)`` interval.

    Returns:
        The overlap fraction in ``[0, 1]``; 0 when ``pred`` is ``None``.
    """
    if pred is None:
        return 0.0
    inter = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union else 0.0


def euler_endpoint_error(steps: int) -> float:
    """Return Euler integration error on a curved unit-time path.

    The reference path has velocity ``v(t) = 3 + pi*cos(pi*t)``, whose exact
    displacement over ``[0, 1]`` is 3. A one-step integrator uses the initial
    velocity for the whole interval and overshoots badly; halving the step
    halves the error. This is the discretization cost that separates a
    many-step diffusion sampler from a few-step flow sampler and a one-step
    distilled model.

    Args:
        steps: Number of equal Euler steps over the unit interval.

    Returns:
        Absolute distance between the integrated and exact endpoints.
    """
    x, dt = 0.0, 1.0 / steps
    for i in range(steps):
        x += (3 + math.pi * math.cos(math.pi * i * dt)) * dt
    return abs(x - 3.0)


for steps in (1, 2, 4, 8, 16):
    print(f"steps={steps:2d}  endpoint error={euler_endpoint_error(steps):.4f}")


import re

INJECTION_PATTERNS = [
    r"ignore (all|previous|the) ",
    r"upload .*(secret|key|token)",
    r"disregard .*instruction",
    r"send .*to https?://",
]


def scan_injection(text: str) -> list[str]:
    """Return the injection patterns an extracted string matches.

    Extracted text — OCR output, a caption, a transcript — is data, never an
    instruction. A non-empty result means the string tried to address the
    agent and must be quarantined; an empty result does not prove safety, it
    only means these patterns did not fire.

    Args:
        text: A string pulled from an image, audio, or document asset.

    Returns:
        The matched patterns, empty when none fire.
    """
    low = text.lower()
    return [p for p in INJECTION_PATTERNS if re.search(p, low)]


import hashlib
import json


def sha256_hex(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def build_manifest(asset: bytes, assertions: list[dict], prev: str | None = None) -> dict:
    """Bind assertions and a hard hash of the asset into a signed manifest.

    The manifest hard-binds to the asset by storing its hash, and chains to a
    prior manifest by storing its digest in ``prev`` — so an edit history is a
    verifiable chain, not a single claim. The signature here is a stand-in
    keyed hash; a real C2PA manifest uses an X.509 credential.

    Args:
        asset: The exact bytes the manifest describes.
        assertions: Creation and edit claims to bind.
        prev: Digest of the previous manifest in an edit chain, if any.

    Returns:
        A manifest dict including its ``sig``.
    """
    body = {"assertions": assertions, "hard_binding": sha256_hex(asset), "prev": prev}
    body["sig"] = sha256_hex((json.dumps(body, sort_keys=True) + "|trusted-key").encode())
    return body


def synthid_detect(asset: bytes, watermarked: bool) -> tuple[bool, float, str]:
    """Return a stubbed watermark detection ``(detected, confidence, scheme)``.

    A positive detection is evidence tied to one scheme and detector; a
    negative is *not* proof of human origin — the asset may be from another
    generator, a stripped version, or an adversarial transform. The stub makes
    that asymmetry explicit to callers.

    Args:
        asset: The asset bytes (unused by the stub; a real detector reads them).
        watermarked: Whether the toy asset carries a soft watermark.

    Returns:
        A ``(detected, confidence, scheme)`` triple.
    """
    return (True, 0.91, "toy-synthid") if watermarked else (False, 0.0, "toy-synthid")


def verify_asset(asset: bytes, manifest: dict | None, watermarked: bool, trusted: bool = True) -> str:
    """Resolve an asset into one of four provenance verdicts.

    The verdicts are deliberately distinct so a product never collapses them
    into "real" and "fake": ``credentials_verified`` (signature valid, hash
    matches, credential trusted), ``validation_failed`` (signature or hash
    broke, or the signer is untrusted), ``watermark_detected`` (no manifest but
    a soft signal survives), and ``no_supported_signal`` (nothing to go on).

    Args:
        asset: The bytes as received, possibly transformed.
        manifest: The manifest, or ``None`` if metadata was stripped.
        watermarked: Whether a soft watermark is present.
        trusted: Whether the signing credential is under a trust policy.

    Returns:
        One of the four verdict strings.
    """
    if manifest is not None:
        body = {k: manifest[k] for k in ("assertions", "hard_binding", "prev")}
        good_sig = sha256_hex((json.dumps(body, sort_keys=True) + "|trusted-key").encode())
        if manifest["sig"] != good_sig or not trusted:
            return "validation_failed"
        if manifest["hard_binding"] != sha256_hex(asset):
            return "validation_failed"
        return "credentials_verified"
    return "watermark_detected" if synthid_detect(asset, watermarked)[0] else "no_supported_signal"
