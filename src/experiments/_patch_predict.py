"""Idempotently expose the batched eval passes in predict.py.

Inserts a `from predict_batched import (...)` block before predict.py's
`if __name__ == "__main__":` guard. Run from the src/ directory:
    python experiments/_patch_predict.py
"""
p = "predict.py"
s = open(p, encoding="utf-8").read()
if "predict_batched" in s:
    print("predict.py: already patched")
else:
    block = (
        "# --- layer8 sweep: batched TF/AR evaluation passes (impl in predict_batched.py) ---\n"
        "try:\n"
        "    from predict_batched import (\n"
        "        regressive_predictions_absolute_batched,\n"
        "        autoregressive_predictions_absolute_batched,\n"
        "        compute_micro_macro,\n"
        "    )\n"
        "except Exception as _e:\n"
        "    pass\n\n\n"
    )
    anchor = 'if __name__ == "__main__":'
    i = s.rfind(anchor)
    s = (s[:i] + block + s[i:]) if i != -1 else (s + "\n\n" + block)
    open(p, "w", encoding="utf-8").write(s)
    print("predict.py: patched")
