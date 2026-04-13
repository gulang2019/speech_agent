This directory tracks the `vllm` delta required by the profiling reproduction flow.

- `mini_vllm/profile/vllm_source.py` pins the upstream repository URL and base commit.
- `mini_vllm/profile/install_and_profile.sh` clones that exact commit into `.deps/` and applies `vllm.patch` if it is non-empty.
- `vllm.patch` is allowed to be whitespace-only. The installer treats that as "no local delta" and skips `git apply`.

To regenerate `vllm.patch` after changing a local `vllm` checkout:

```bash
git -C /path/to/vllm diff --binary <base-commit> > mini_vllm/profile/patches/vllm.patch
```
