# Changelog

## 2026-04-21

### `request_utils.py`

#### Cleanup
- Removed unused imports (`from os import name`, `import heapq`) and dead global `time_ratio`
- Removed `_target_model_idx` field from `PipelineRequest` (was set in `add_request` but never read)
- Removed dead commented-out code block in `_continuous_batching_scheduler` (old mixed prefill+decode batching logic)
- Removed dead `expected_time` variable in `send_batch`
- Renamed `send_batch` → `_send_batch` (private convention, consistent with `_continuous_batching_scheduler`)

#### vLLM Consistency Fixes (behavior changes)
- **`query_lens` in Prefill**: was incorrectly set to `req.target_generation_length` (always 0 for new requests); fixed to `req.original_input_len` — prefill time now correctly scales with actual input length
- **`context_lens` in Decode**: was only `generated_tokens_count`; fixed to `original_input_len + generated_tokens_count` — matches vLLM's KV-cache context model (total context = prompt + generated tokens)
- Removed the line that overwrote `req.original_input_len` with `batch_metadata.context_lens[i]` (which is 0 for prefill) during `process_batch`; the original input length is now preserved throughout the request lifecycle

#### Refactoring
- Added `RequestStatus` enum (`PREFILL`, `DECODE`) replacing `"Prefill"`/`"Decode"` string literals throughout
- Moved `time_delay_prefill` and `time_delay_decode` from module-level globals to `AsyncPipelineEngine.__init__` parameters (defaults: `2.0` and `0.02` respectively)
- Simplified `run()` post-processing: replaced an O(n²) list-comprehension filter (using repeated `req_id not in [...]`) with an O(n) set lookup; unified "re-admit or complete" logic into one clear loop
- Fixed TTFF/TBF calculation bug in `main()`: `TTNF` elements are `(timestamp, model_idx)` tuples — now correctly extracts the timestamp component before computing differences

---

### `dataloader.py`

- Fixed `prepare_data(dataset)` to use its `dataset` parameter instead of the global `MultiD` (the function parameter was previously ignored)
- Renamed `list` → `values` in `violation_calc` (shadowed the Python builtin `list`)
- Removed dead loop over `completed_user_data` in `single_task` (the list was always empty because the corresponding `append` call was commented out)
