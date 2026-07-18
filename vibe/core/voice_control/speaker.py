from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from vibe.core.audio_player.audio_player_port import AudioFormat
from vibe.core.logger import logger

if TYPE_CHECKING:
    from vibe.core.audio_player.audio_player_port import AudioPlayerPort
    from vibe.core.tts.tts_client_port import TTSClientPort


class SpeakerPort(Protocol):
    @property
    def is_speaking(self) -> bool: ...

    async def speak(self, text: str) -> None: ...

    def stop(self) -> None: ...

    async def close(self) -> None: ...


class Speaker:
    """Synthesizes text with the TTS client and plays it, blocking until done."""

    def __init__(
        self, tts_client: TTSClientPort, audio_player: AudioPlayerPort
    ) -> None:
        self._tts = tts_client
        self._player = audio_player

    @property
    def is_speaking(self) -> bool:
        return self._player.is_playing

    async def speak(self, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        try:
            result = await self._tts.speak(clean)
        except Exception:
            logger.error("TTS synthesis failed", exc_info=True)
            return
        if not result.audio_data:
            return
        loop = asyncio.get_running_loop()
        done = asyncio.Event()
        self._player.stop()
        self._player.play(
            result.audio_data,
            AudioFormat.WAV,
            on_finished=lambda: loop.call_soon_threadsafe(done.set),
        )
        await done.wait()

    def stop(self) -> None:
        self._player.stop()

    async def close(self) -> None:
        self._player.stop()
        await self._tts.close()
