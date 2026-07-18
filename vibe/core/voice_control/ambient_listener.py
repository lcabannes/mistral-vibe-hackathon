from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vibe.core.audio_recorder.audio_recorder_port import RecordingMode
from vibe.core.logger import logger
from vibe.core.transcribe.transcribe_client_port import (
    TranscribeDone,
    TranscribeError,
    TranscribeSessionCreated,
    TranscribeTextDelta,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from vibe.core.audio_recorder.audio_recorder_port import AudioRecorderPort
    from vibe.core.transcribe.transcribe_client_port import (
        TranscribeClientPort,
        TranscribeEvent,
    )


@dataclass(frozen=True, slots=True)
class AmbientCallbacks:
    on_delta: Callable[[str], None]
    on_listening: Callable[[], None] | None = None
    on_error: Callable[[str], None] | None = None


class AmbientListener:
    """Continuously streams the mic through the transcribe client, forwarding
    each transcript delta. Reconnects the session on any transport failure.
    """

    def __init__(
        self,
        audio_recorder: AudioRecorderPort,
        client_factory: Callable[[], TranscribeClientPort],
        *,
        sample_rate: int,
        callbacks: AmbientCallbacks,
        reconnect_delay: float = 0.4,
    ) -> None:
        self._audio_recorder = audio_recorder
        self._client_factory = client_factory
        self._sample_rate = sample_rate
        self._cb = callbacks
        self._reconnect_delay = reconnect_delay
        self._running = False
        self._task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._audio_recorder.cancel()

    async def _run(self) -> None:
        while self._running:
            client = self._client_factory()
            try:
                self._audio_recorder.start(
                    RecordingMode.STREAM, sample_rate=self._sample_rate
                )
                await self._process_session(client, self._audio_recorder.audio_stream())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Ambient listening session failed", exc_info=True)
                if self._cb.on_error is not None:
                    self._cb.on_error(str(exc))
            finally:
                self._audio_recorder.cancel()
                await client.close()
            if self._running:
                await asyncio.sleep(self._reconnect_delay)

    async def _process_session(
        self, client: TranscribeClientPort, audio_stream: AsyncIterator[bytes]
    ) -> None:
        async for event in client.transcribe(audio_stream):
            if not self._handle_event(event):
                return

    def _handle_event(self, event: TranscribeEvent) -> bool:
        match event:
            case TranscribeTextDelta(text=text):
                self._cb.on_delta(text)
                return True
            case TranscribeSessionCreated():
                if self._cb.on_listening is not None:
                    self._cb.on_listening()
                return True
            case TranscribeDone() | TranscribeError():
                return False
