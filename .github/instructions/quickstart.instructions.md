## Quickstart code testing

The quickstart code shown in `README.md` and `docs/code_example.md` is tested automatically via `src/multiview_stitcher/_tests/test_quickstart.py`. When modifying any quickstart-related code, follow these rules:

### How the tests work

1. **Canonical source of truth** — The actual executable code lives in numbered `.py` files under `src/multiview_stitcher/_tests/quickstart/`:
   - `0_data_preparation.py`
   - `1_registration.py`
   - `2_fusion.py`
   - `3_fusion_large.py`

2. **`test_quickstart_code`** — Executes all `quickstart/*.py` files in sorted order within a shared `globals()` namespace (so later blocks can reference variables from earlier ones).

3. **`test_quickstart_code_sync`** — Asserts that the exact text of every `quickstart/*.py` file appears verbatim inside **both** `README.md` and `docs/code_example.md`. This test fails if the markdown docs fall out of sync with the source files.

### Rules for changes

- **When editing the markdown docs or README.md also edit the canonical `.py` files**, and vice versa. Both must stay identical or `test_quickstart_code_sync` will fail.