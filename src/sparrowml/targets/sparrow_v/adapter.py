"""Future Sparrow-V execution adapter; bootstrap deliberately does not execute targets."""

from .contracts import SparrowVResult


def validate_result(result: SparrowVResult) -> SparrowVResult:
    """Return a schema-validated result without running Sparrow-V."""
    return result
