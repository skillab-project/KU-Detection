"""
Per-file analyzer for the Detector17 model.

Mirrors the shape of core.analysis.codebert_sliding_window.codebert_sliding_window
so routes.py can swap pipelines based on ku_source.

Difference from legacy: the Detector17 model operates on the whole file (truncated
to 4096 chars). No sliding window — the model was trained on file-level snippets,
and sliding would multiply noise without improving recall.
"""

from typing import List
from core.utils.code_file import CodeFile
from core.ml_operations.detector17_loader import Detector17Model


def detector17_file_scan(files: List[CodeFile], model: Detector17Model) -> dict:
    """
    Runs Detector17 inference on each file. Returns a dict:
        {filename: {ku_id: 0_or_1}}

    Also calls f.add_ku_result(ku_id, value) on each CodeFile so the
    data flows into save_analysis_to_db() identically to the legacy path.
    """
    file_results = {}
    for f in files:
        code = "\n".join(f.lines) if hasattr(f, "lines") else getattr(f, "content", "")
        if not code or not code.strip():
            # Empty file → all zeros
            detected = {ku: 0 for ku in model.predict("placeholder").keys()}
        else:
            detected = model.predict_binary(code)

        file_results[f.filename] = detected
        for ku_id, val in detected.items():
            f.add_ku_result(ku_id, val)

    return file_results
