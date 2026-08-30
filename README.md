# Reasoning Controls Are Not a Common Compute Axis: A Three-Model BBQ Audit 
Neurips 2026 TAE Submission

Abstract: Test-time reasoning controls are often treated as a single compute axis, although their interfaces and effects differ across backends. We audit Qwen3.5-9B, GPT-OSS-20B, and Kimi-K2.6 on 1,570 BBQ items, comprising 1,020 disambiguated and 550 ambiguous items. Natural-output accuracy is primary. Recovered accuracy is a sensitivity analysis that reuses the terminal prefix-probe argmax only when a natural final answer is missing. Qwen native effort settings increased disambiguated accuracy by 2.84 to 3.92 percentage points relative to native off, whereas GPT low effort reduced it by 3.24 points and Kimi reasoning on reduced it by 5.39 points. At a 512-token cap, no-final rates reached 65.67% for Qwen and 21.78% for Kimi, so the corresponding accuracy losses mix answer quality with noncompletion. Most gold-centered directional contrasts included zero. The two shortest-cap exceptions coincided with high noncompletion, while raw directional scores near $-10$ matched the selected sample's gold baseline of $-10$. All verified-pair gap intervals included zero. Together, these results show that reasoning controls are backend-specific interventions rather than a shared compute axis; their evaluation should report natural completion separately from probe-based recovery and interpret directional scores against the evaluated sample's gold baseline.

How to run:

Place specified model in config with endpoint url, api key, model id, and other configurations before running

python main.py for running with a random generated dataset

python main.py --jsonl file.jsonl for running with a specified dataset

python metrics.py for metrics on the results
