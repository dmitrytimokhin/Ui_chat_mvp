import logging
from typing import Sequence


class HealthcheckFilter(logging.Filter):
    """Suppress logs that originate from Kubernetes health/readiness probes.

    It looks for common probe paths in the log message and drops those records.
    """

    def __init__(self, paths: Sequence[str] | None = None):
        super().__init__()
        if paths is None:
            paths = (
                "/helthz/liveness",
                "/helthz/readiness",
                "/healthz/liveness",
                "/healthz/readiness",
                "/healthz/ready",
                "/healthz/live",
            )
        self.paths = tuple(paths)

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple filter
        try:
            msg = record.getMessage()
        except Exception as e:
            return True

        for p in self.paths:
            if p in msg:
                return False
        return True
