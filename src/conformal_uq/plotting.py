"""Reserved plotting module; scientific figures begin only after analysis approval."""


def require_stage12() -> None:
    raise RuntimeError("Scientific plotting is intentionally deferred to Stage 12.")
