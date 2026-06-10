"""Memory-mapped token corpus for large pretraining datasets.

A tokenized corpus is stored as a flat ``uint16`` ``.bin`` of token ids plus a
sidecar ``.json`` (``{n_tokens, vocab_size, dtype}``). :class:`MemmapTokenCorpus`
exposes the ``TokenSource`` slicing interface (``numel`` + ``__getitem__``) so
``spikegpt.language.sample_token_batch`` can draw IID random windows from corpora
far larger than RAM (e.g. OpenWebText2) without re-tokenizing each run. ``uint16``
covers vocabularies up to 65535 (GPT-NeoX is 50277).
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = "uint16"


class MemmapTokenCorpus:
    """Read-only memory-mapped view over a tokenized ``.bin`` corpus."""

    def __init__(
        self,
        bin_path: str | PathLike[str],
        *,
        n_tokens: int,
        vocab_size: int,
        dtype: str = TOKEN_DTYPE,
    ) -> None:
        self.path = Path(bin_path)
        self.n_tokens = int(n_tokens)
        self.vocab_size = int(vocab_size)
        self.dtype = np.dtype(dtype)
        self._mm = np.memmap(self.path, dtype=self.dtype, mode="r", shape=(self.n_tokens,))

    @classmethod
    def open(cls, bin_path: str | PathLike[str]) -> MemmapTokenCorpus:
        """Open a corpus written by :class:`TokenCorpusWriter` (reads the sidecar)."""
        path = Path(bin_path)
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(
            path,
            n_tokens=int(meta["n_tokens"]),
            vocab_size=int(meta["vocab_size"]),
            dtype=str(meta.get("dtype", TOKEN_DTYPE)),
        )

    def numel(self) -> int:
        return self.n_tokens

    def __len__(self) -> int:
        return self.n_tokens

    def __getitem__(self, index: slice) -> torch.Tensor:
        # Copy the (uint16) window into int64 so it round-trips through torch.long
        # exactly and is independent of the underlying read-only memmap buffer.
        return torch.from_numpy(np.asarray(self._mm[index], dtype=np.int64))

    def split_tail(self, tail_tokens: int) -> tuple[TokenArrayView, TokenArrayView]:
        """Return ``(head, tail)`` no-copy views: the last ``tail_tokens`` as the
        tail (an in-domain held-out validation slice), the rest as the head."""
        if not 0 < tail_tokens < self.n_tokens:
            raise ValueError(f"tail_tokens must be in (0, {self.n_tokens}); got {tail_tokens}")
        cut = self.n_tokens - tail_tokens
        return (
            TokenArrayView(self._mm[:cut], self.vocab_size),
            TokenArrayView(self._mm[cut:], self.vocab_size),
        )


class TokenArrayView:
    """A contiguous, no-copy sub-range of a token array exposing ``TokenSource``."""

    def __init__(self, array: np.ndarray, vocab_size: int) -> None:
        self._array = array
        self.vocab_size = int(vocab_size)
        self.n_tokens = int(len(array))

    def numel(self) -> int:
        return self.n_tokens

    def __len__(self) -> int:
        return self.n_tokens

    def __getitem__(self, index: slice) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self._array[index], dtype=np.int64))


class TokenCorpusWriter:
    """Stream token ids to a ``uint16`` ``.bin`` and write the sidecar on close."""

    def __init__(
        self,
        bin_path: str | PathLike[str],
        *,
        vocab_size: int,
        dtype: str = TOKEN_DTYPE,
    ) -> None:
        self.dtype = np.dtype(dtype)
        max_id = np.iinfo(self.dtype).max
        if vocab_size - 1 > max_id:
            raise ValueError(f"vocab_size {vocab_size} exceeds dtype {dtype} range (max {max_id})")
        self.path = Path(bin_path)
        self.vocab_size = int(vocab_size)
        self.n_tokens = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("wb")

    def write(self, ids: np.ndarray | list[int]) -> None:
        array = np.asarray(ids, dtype=self.dtype)
        self._handle.write(array.tobytes())
        self.n_tokens += int(array.size)

    def flush_sidecar(self) -> None:
        """Flush the .bin and (re)write the .json for the current token count.

        Calling this periodically makes a partially-written corpus usable even if
        the producer (e.g. a streaming HF dataset) dies mid-run — the sidecar
        always describes a valid prefix of the .bin.
        """
        self._handle.flush()
        self.path.with_suffix(".json").write_text(
            json.dumps(
                {"n_tokens": self.n_tokens, "vocab_size": self.vocab_size, "dtype": self.dtype.name}
            )
        )

    def close(self) -> None:
        self._handle.close()
        self.path.with_suffix(".json").write_text(
            json.dumps(
                {"n_tokens": self.n_tokens, "vocab_size": self.vocab_size, "dtype": self.dtype.name}
            )
        )

    def __enter__(self) -> TokenCorpusWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
