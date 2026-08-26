# Source and rewrite note

- Repository: <https://github.com/fkchaos/a-share-quant-sim>
- Pinned commit: `47d789c4f3e8977f755fcb9e492d452f49d7c8b8`
- Original files:
  - <https://github.com/fkchaos/a-share-quant-sim/blob/47d789c4f3e8977f755fcb9e492d452f49d7c8b8/core/earnings_preview.py>
  - <https://github.com/fkchaos/a-share-quant-sim/blob/47d789c4f3e8977f755fcb9e492d452f49d7c8b8/core/event_factors.py>
- License: MIT, copyright 2026 OWL / ZOO. License text: <https://github.com/fkchaos/a-share-quant-sim/blob/47d789c4f3e8977f755fcb9e492d452f49d7c8b8/LICENSE>

`event_formulas.py` is an adapted formula reference. It removes SQLite, local database paths, download hooks, `datetime.now()`, simulator objects, normalization against future/full samples, and all strategy adapters. It adds explicit `available_at <= inference_at` filtering and uses this project's event field names.

The earnings helper still expects `is_positive`/`is_negative`; a Fold must derive those flags from visible `forecast_vip` `type` and `p_change_min`/`p_change_max` after inspecting actual values. The holder helper assumes observed `in_de` values `IN`/`DE`; verify them before use. A mismatch must fail or exclude rows, not invent a mapping.

Fold rewrite: copy no database or framework code. Re-express only the selected arithmetic in the self-contained `output/main.py` and combine it with this project's daily tradability and strict order contract. Never import this reference file.

## MIT license notice

Copyright (c) 2026 OWL / ZOO

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
