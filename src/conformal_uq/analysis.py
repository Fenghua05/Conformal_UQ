"""Reserved analysis module; analysis begins after locked experiment results exist."""


def require_stage11() -> None:
    raise RuntimeError("Statistical analysis is intentionally deferred to Stage 11.")
