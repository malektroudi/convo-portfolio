"""Energy endpointing: deciding the user has stopped talking.

This is day 2's rejection threshold and day 10's grounding threshold wearing a
third costume, and the shape is identical: one number, two failure modes
pulling in opposite directions, and no setting that avoids both.

  hangover too short   the turn is cut off mid-sentence, because people pause
                       inside sentences and a pause is indistinguishable from
                       an ending until more audio arrives
  hangover too long    the person sits in silence waiting, and every
                       millisecond is spent computing nothing at all

What makes endpointing the *worst* of the three is that its cost is paid in
pure latency. A classifier threshold trades accuracy for accuracy; this one
trades accuracy for dead air. It is also the only stage whose correct value is
a product decision — a phone line for emergency calls and a drive-through
order do not want the same number — which is why the build sweeps it and the
tab exposes it rather than either one picking a default and calling it tuned.

Energy, not a neural VAD, deliberately: it is free, it runs in microseconds,
and on clean input the endpointing *error* comes from the hangover rule rather
than from frame classification. On a noisy line that stops being true, and
this would need a real VAD — said here rather than discovered later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 20
"""Frame size for energy. 20 ms is the usual choice and divides a 16 kHz rate."""

SILENCE_FLOOR = 0.012
"""RMS below which a frame counts as silence, on audio normalised to +/-1."""


def frames(audio: np.ndarray, frame_ms: int = FRAME_MS) -> np.ndarray:
    """Reshape to (n_frames, frame_len), dropping any ragged tail."""
    size = int(SAMPLE_RATE * frame_ms / 1000)
    usable = len(audio) - (len(audio) % size)
    if usable <= 0:
        return np.zeros((0, size), dtype=np.float32)
    return audio[:usable].reshape(-1, size)


def frame_energy(audio: np.ndarray, frame_ms: int = FRAME_MS) -> np.ndarray:
    """RMS per frame."""
    blocks = frames(audio, frame_ms)
    if not len(blocks):
        return np.zeros(0, dtype=np.float32)
    return np.sqrt(np.mean(np.square(blocks), axis=1))


def speech_frames(
    audio: np.ndarray, floor: float = SILENCE_FLOOR, frame_ms: int = FRAME_MS
) -> np.ndarray:
    return frame_energy(audio, frame_ms) >= floor


@dataclass(frozen=True)
class Endpoint:
    """Where a hangover rule decided the turn ended."""

    endpoint_ms: float
    """Offset into the audio at which the turn was called finished."""
    last_speech_ms: float
    """Offset of the final speech frame — the true end of the utterance."""
    audio_ms: float
    truncated: bool
    """True when the endpoint landed before the final speech frame."""
    hangover_ms: float

    @property
    def wait_ms(self) -> float:
        """Dead air: endpoint minus the moment the person actually stopped.

        Negative when the rule fired early, which is not a latency saving —
        it is the user being interrupted, and `truncated` is how that is
        counted.
        """
        return round(self.endpoint_ms - self.last_speech_ms, 2)

    @property
    def kept_samples(self) -> int:
        return int(self.endpoint_ms * SAMPLE_RATE / 1000)


def endpoint(
    audio: np.ndarray,
    hangover_ms: float,
    floor: float = SILENCE_FLOOR,
    frame_ms: int = FRAME_MS,
) -> Endpoint:
    """Find where a given hangover would end the turn.

    Walks the frames once, exactly as a live endpointer would: a run of
    silence at least `hangover_ms` long ends the turn there and then, with no
    knowledge of what comes after. That matters — a rule allowed to look ahead
    would never cut anyone off, and would not be implementable.
    """
    is_speech = speech_frames(audio, floor, frame_ms)
    audio_ms = len(audio) / SAMPLE_RATE * 1000
    needed = max(1, int(round(hangover_ms / frame_ms)))

    last_speech = int(np.max(np.nonzero(is_speech)[0])) if is_speech.any() else -1
    last_speech_ms = (last_speech + 1) * frame_ms if last_speech >= 0 else 0.0

    run = 0
    started = False
    for index, speaking in enumerate(is_speech):
        if speaking:
            started, run = True, 0
            continue
        run += 1
        # Silence before any speech is not an ending, it is someone thinking.
        if started and run >= needed:
            fired_ms = (index + 1) * frame_ms
            return Endpoint(
                endpoint_ms=float(fired_ms),
                last_speech_ms=float(last_speech_ms),
                audio_ms=float(audio_ms),
                truncated=fired_ms < last_speech_ms,
                hangover_ms=float(hangover_ms),
            )

    # Never fired: the audio ran out first, so the turn ends with the buffer.
    return Endpoint(
        endpoint_ms=float(audio_ms),
        last_speech_ms=float(last_speech_ms),
        audio_ms=float(audio_ms),
        truncated=False,
        hangover_ms=float(hangover_ms),
    )


def longest_internal_pause_ms(
    audio: np.ndarray, floor: float = SILENCE_FLOOR, frame_ms: int = FRAME_MS
) -> float:
    """The longest silence *between* speech, which is what cuts people off.

    Any hangover shorter than this on a given utterance ends the turn while
    the person is still mid-sentence. It is the floor under the endpointing
    target, and it is a property of how people talk rather than of the system.
    """
    is_speech = speech_frames(audio, floor, frame_ms)
    if not is_speech.any():
        return 0.0
    first = int(np.min(np.nonzero(is_speech)[0]))
    last = int(np.max(np.nonzero(is_speech)[0]))
    longest = run = 0
    for speaking in is_speech[first : last + 1]:
        run = 0 if speaking else run + 1
        longest = max(longest, run)
    return float(longest * frame_ms)
