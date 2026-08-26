# Source and rewrite note

- Repository: <https://github.com/crisq-star/alpha-lab-cn>
- Pinned commit: `44f599819d1735170f3730b8bc98623b8d22036b`
- Original files:
  - <https://github.com/crisq-star/alpha-lab-cn/blob/44f599819d1735170f3730b8bc98623b8d22036b/src/factors/momentum.py>
  - <https://github.com/crisq-star/alpha-lab-cn/blob/44f599819d1735170f3730b8bc98623b8d22036b/src/factors/value.py>
  - <https://github.com/crisq-star/alpha-lab-cn/blob/44f599819d1735170f3730b8bc98623b8d22036b/src/factors/sentiment.py>
- License: MIT, copyright 2024 Alpha Lab CN. License text: <https://github.com/crisq-star/alpha-lab-cn/blob/44f599819d1735170f3730b8bc98623b8d22036b/LICENSE>

`formulas.py` is an adapted, framework-free excerpt. It omits the repository's factor base classes, neutralization pipeline, data acquisition, forward-return analysis, ML, LLM, dashboard, and reported results. It also makes pandas return behavior explicit and keeps only NumPy/pandas arithmetic.

Fold rewrite: map `close`, `adj_factor`, `vol`, `turnover_rate`, and `pe_ttm` from the current T-1 `daily` view, then place the chosen formula directly in `output/main.py`. Never import this reference file.

## MIT license notice

Copyright (c) 2024 Alpha Lab CN

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
