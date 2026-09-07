"""Agent Release Impact Gate API, retaining its historical Python import namespace.

Research exports load only when requested, so the release CLI does not import the
optional scientific stack. Existing Python callers keep the same public names.
"""

from importlib import import_module

_EXPORTS = {
    "ActionGateway": "action_gateway",
    "AuditStore": "action_gateway",
    "generate_benchmark": "benchmark",
    "compute_metrics": "evaluation",
    "production_score": "evaluation",
    "run_experiment": "simulator",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "ActionGateway",
    "AuditStore",
    "compute_metrics",
    "generate_benchmark",
    "production_score",
    "run_experiment",
]
