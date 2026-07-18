from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
from typing import Protocol, cast

from vibe.core.utils.platform import is_windows


class _FcntlModule(Protocol):
    LOCK_EX: int
    LOCK_UN: int

    def flock(self, fd: int, operation: int) -> None: ...


class _MsvcrtModule(Protocol):
    LK_LOCK: int
    LK_UNLCK: int

    def locking(self, fd: int, mode: int, nbytes: int) -> None: ...


@contextmanager
def process_file_lock(path: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT
    if no_follow := getattr(os, "O_NOFOLLOW", 0):
        flags |= no_follow
    if path.is_symlink():
        raise OSError(f"Unsafe lock path: {path}")
    fd = os.open(path, flags, 0o600)
    try:
        _lock(fd)
        yield
    finally:
        try:
            _unlock(fd)
        finally:
            os.close(fd)


def _lock(fd: int) -> None:
    if is_windows():
        module = cast(_MsvcrtModule, importlib.import_module("msvcrt"))
        _ensure_lock_byte(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        module.locking(fd, module.LK_LOCK, 1)
        return
    module = cast(_FcntlModule, importlib.import_module("fcntl"))
    module.flock(fd, module.LOCK_EX)


def _unlock(fd: int) -> None:
    if is_windows():
        module = cast(_MsvcrtModule, importlib.import_module("msvcrt"))
        os.lseek(fd, 0, os.SEEK_SET)
        module.locking(fd, module.LK_UNLCK, 1)
        return
    module = cast(_FcntlModule, importlib.import_module("fcntl"))
    module.flock(fd, module.LOCK_UN)


def _ensure_lock_byte(fd: int) -> None:
    if os.fstat(fd).st_size:
        return
    os.write(fd, b"\0")
    os.fsync(fd)
