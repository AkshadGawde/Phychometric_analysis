"""Code shared by the research pipeline (src/02 - src/05) and the live app (src/predict.py).

Everything that turns a raw input into model features lives here, so training and the app can
never drift apart. The numbered scripts import from these modules; they add only the file
handling (paths, caching, logs) around them.
"""
