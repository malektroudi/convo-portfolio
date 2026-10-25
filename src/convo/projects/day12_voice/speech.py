"""STT and TTS, both real, both local, both free.

    uv sync --extra train --extra voice

Imports are inside the functions on purpose. `onnxruntime` and `torch` are
build-time dependencies here exactly as `torch` is for every other day: the
hub reads committed timings and plays a committed wav, so neither ever
reaches the serving image. A test asserts that this module is importable
without either installed, because a tab that crashes on import takes the whole
hub down with it.

The models are the small ones, and that is a latency decision rather than a
concession. `whisper-tiny.en` is 38M parameters and `en_US-lessac-low` is the
low-quality Piper voice; on a budget measured in hundreds of milliseconds, a
better model that takes a second longer is not a better model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from convo.core import get_logger

STT_MODEL = "openai/whisper-tiny.en"
TTS_VOICE = "en_US-lessac-low"
TTS_REPO = "rhasspy/piper-voices"
TTS_PATH = "en/en_US/lessac/low/en_US-lessac-low"
SAMPLE_RATE = 16000

log = get_logger("day12.speech")


@dataclass
class Transcriber:
    """whisper-tiny.en on CPU, greedy, so a run is reproducible."""

    model_id: str = STT_MODEL
    _model: Any = None
    _processor: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        started = time.perf_counter()
        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.float32
        ).eval()
        log.info(
            "stt loaded",
            extra={
                "model": self.model_id,
                "seconds": round(time.perf_counter() - started, 1),
                "params_m": round(sum(p.numel() for p in self._model.parameters()) / 1e6),
            },
        )

    def transcribe(self, audio: np.ndarray, max_new_tokens: int = 64) -> tuple[str, float]:
        """Return (text, milliseconds). Audio is float32 mono at 16 kHz."""
        import torch

        self.load()
        started = time.perf_counter()
        features = self._processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        with torch.no_grad():
            ids = self._model.generate(features, max_new_tokens=max_new_tokens)
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip(), (time.perf_counter() - started) * 1000


@dataclass
class AudioPiece:
    """One synthesised sentence, with when it became available."""

    text: str
    pcm: bytes
    offset_ms: float
    """Milliseconds from the start of synthesis until this piece was ready."""

    @property
    def duration_ms(self) -> float:
        return len(self.pcm) / (SAMPLE_RATE * 2) * 1000


@dataclass
class Speaker:
    """Piper, which synthesises one sentence at a time.

    That granularity is the whole reason streaming works here: Piper will not
    give you half a sentence, but it will give you the first sentence of a
    four-sentence reply without waiting for the other three — so the critical
    path is the length of the first sentence, not of the answer.
    """

    voice_name: str = TTS_VOICE
    deterministic: bool = True
    """Zero the sampling noise so two builds produce identical audio.

    Piper is a VITS-style model: it samples from a latent distribution, so the
    same sentence synthesised twice gives different audio — different enough
    that whisper transcribed one of these questions correctly on one run and
    as "Our Assistant Stogs allowed" on the next. Word error rate measured
    over a non-reproducible corpus is not a measurement, it is a draw.

    The cost is that every utterance is read in the flattest possible version
    of the voice, with none of the prosodic variation the noise exists to
    provide. For measuring latency that costs nothing. For measuring *speech
    recognition* it makes the input even less like a real caller than it
    already was, which is the larger of the two caveats on the WER here.
    """
    _voice: Any = None
    _config: Any = None
    _rate: int = field(default=SAMPLE_RATE)

    def load(self) -> None:
        if self._voice is not None:
            return
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice, SynthesisConfig

        started = time.perf_counter()
        model = hf_hub_download(TTS_REPO, f"{TTS_PATH}.onnx")
        config = hf_hub_download(TTS_REPO, f"{TTS_PATH}.onnx.json")
        self._voice = PiperVoice.load(model, config_path=config)
        self._config = (
            SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0, length_scale=1.0)
            if self.deterministic
            else None
        )
        log.info(
            "tts loaded",
            extra={"voice": self.voice_name, "seconds": round(time.perf_counter() - started, 1)},
        )

    def synthesize(self, text: str) -> list[AudioPiece]:
        """Synthesise `text`, recording when each sentence became available."""
        self.load()
        started = time.perf_counter()
        pieces: list[AudioPiece] = []
        for chunk in self._voice.synthesize(text, syn_config=self._config):
            self._rate = chunk.sample_rate
            pieces.append(
                AudioPiece(
                    text="",
                    pcm=chunk.audio_int16_bytes,
                    offset_ms=(time.perf_counter() - started) * 1000,
                )
            )
        return pieces

    def first_audio_ms(self, text: str) -> tuple[float, list[AudioPiece]]:
        """Milliseconds until the first sample of `text` is ready to play."""
        pieces = self.synthesize(text)
        return (pieces[0].offset_ms if pieces else 0.0), pieces


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def write_wav(path, audio: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """Write a mono 16-bit wav with the standard library, no soundfile needed."""
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(float_to_pcm(audio))


def pad_with_silence(audio: np.ndarray, lead_ms: float = 150, tail_ms: float = 1200) -> np.ndarray:
    """Surround an utterance with silence.

    The tail is what an endpointer actually consumes — synthesised speech stops
    dead on the last phoneme, and a buffer that ends exactly when the speaking
    does would let every hangover setting look identical because the audio runs
    out before any of them fire.
    """
    lead = np.zeros(int(SAMPLE_RATE * lead_ms / 1000), dtype=np.float32)
    tail = np.zeros(int(SAMPLE_RATE * tail_ms / 1000), dtype=np.float32)
    return np.concatenate([lead, audio.astype(np.float32), tail])


def normalise(text: str) -> list[str]:
    """Words, lowercased, with punctuation dropped — for scoring a transcript.

    Whisper writes "How long do you hold my table if I am late?" where the
    reference says "how long do you hold my table if i am late". Counting the
    capital and the question mark as errors would measure formatting, not
    recognition.
    """
    import re

    return re.findall(r"[a-z0-9']+", text.lower())


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: edit distance over words, divided by reference length.

    Returned uncapped. A transcript longer than the reference can score above
    1.0, and clamping it would hide the case this day cares about — an
    endpointer that fires early produces a *short* transcript, and one that
    never fires lets silence become hallucinated words.
    """
    want, got = normalise(reference), normalise(hypothesis)
    if not want:
        return 0.0 if not got else 1.0

    previous = list(range(len(got) + 1))
    for i, want_word in enumerate(want, start=1):
        current = [i]
        for j, got_word in enumerate(got, start=1):
            current.append(
                previous[j - 1]
                if want_word == got_word
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return round(previous[-1] / len(want), 4)
