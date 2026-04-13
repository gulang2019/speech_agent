from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class VllmSourceSpec:
    repo_url: str
    commit: str
    patch_path: Path

    @property
    def short_commit(self) -> str:
        return self.commit[:12]

    @property
    def patch_bytes(self) -> bytes:
        if not self.patch_path.exists():
            return b""
        return self.patch_path.read_bytes()

    @property
    def patch_is_nonempty(self) -> bool:
        return bool(self.patch_bytes.strip())

    @property
    def patch_sha256(self) -> str | None:
        if not self.patch_is_nonempty:
            return None
        return sha256(self.patch_bytes).hexdigest()

    @property
    def managed_checkout_dirname(self) -> str:
        patch_suffix = self.patch_sha256[:12] if self.patch_sha256 else "nopatch"
        return f"vllm-{self.short_commit}-{patch_suffix}"


VLLM_SOURCE = VllmSourceSpec(
    repo_url="https://github.com/vllm-project/vllm.git",
    commit="e1cd7a5faffd188cd204f7b54eea6cb35f787ee9",
    patch_path=_THIS_DIR / "patches" / "vllm.patch",
)
