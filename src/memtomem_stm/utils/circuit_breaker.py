"""Unified circuit breaker for STM subsystems."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Three-state circuit breaker: closed → open → half-open.

    - **closed**: all calls pass through.
    - **open**: all calls blocked; transitions to half-open after ``reset_timeout``.
    - **half-open**: one probe call allowed; success closes, failure re-opens.
    """

    def __init__(
        self,
        max_failures: int = 3,
        reset_timeout: float = 60.0,
        name: str = "",
    ) -> None:
        self._max_failures = max_failures
        self._reset_timeout = reset_timeout
        self._name = name
        self._failures = 0
        self._state = "closed"
        self._opened_at = 0.0

    @property
    def is_open(self) -> bool:
        if self._state == "closed":
            return False
        if self._state == "open" and time.monotonic() - self._opened_at >= self._reset_timeout:
            self._state = "half-open"
            return False  # allow one probe
        return self._state == "open"

    @property
    def state(self) -> str:
        """Current state: 'closed', 'open', or 'half-open'."""
        # Trigger half-open transition if timeout elapsed
        if self._state == "open" and time.monotonic() - self._opened_at >= self._reset_timeout:
            self._state = "half-open"
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failures

    @property
    def time_until_reset(self) -> float | None:
        """Seconds until open breaker transitions to half-open. None if not open."""
        if self._state != "open":
            return None
        remaining = self._reset_timeout - (time.monotonic() - self._opened_at)
        return max(0.0, remaining)

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failures += 1
        if self._state == "half-open" or (
            self._failures >= self._max_failures and self._state == "closed"
        ):
            self._state = "open"
            self._opened_at = time.monotonic()
            logger.warning(
                "CircuitBreaker[%s] opened after %d failures", self._name, self._failures
            )

    # Aliases for backward compatibility
    success = record_success
    failure = record_failure

class TestCircuitBreakerProperties:
    def test_time_until_reset_when_closed(self):
        cb = CircuitBreaker()
        assert cb.time_until_reset is None

    def test_time_until_reset_when_open(self):
        cb = CircuitBreaker(max_failures=1, reset_timeout=60.0)
        cb.record_failure()
        remaining = cb.time_until_reset
        assert remaining is not None
        assert 0 < remaining <= 60.0

    def test_time_until_reset_when_half_open(self):
        cb = CircuitBreaker(max_failures=1, reset_timeout=10.0)
        cb.record_failure()

        with patch("memtomem_stm.utils.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = cb._opened_at + 11.0
            assert not cb.is_open  # triggers half-open transition

        # Now in half-open — time_until_reset should be None
        assert cb.time_until_reset is None

    def test_time_until_reset_at_boundary(self):
        cb = CircuitBreaker(max_failures=1, reset_timeout=10.0)
        cb.record_failure()

        with patch("memtomem_stm.utils.circuit_breaker.time") as mock_time:
            # exactly at boundary
            mock_time.monotonic.return_value = cb._opened_at + 10.0
            result = cb.time_until_reset
            assert result == 0.0

    def test_backward_compat_failure_alias(self):
        cb = CircuitBreaker(max_failures=2)
        cb.failure()  # alias for record_failure
        assert cb.failure_count == 1

    def test_backward_compat_success_alias(self):
        cb = CircuitBreaker(max_failures=2)
        cb.failure()
        cb.failure()
        assert cb.is_open
        cb.success()  # alias for record_success
        assert cb.failure_count == 0
        assert cb._state == "closed"

    def test_backward_compat_aliases_match_originals(self):
        assert CircuitBreaker.success == CircuitBreaker.record_success
        assert CircuitBreaker.failure == CircuitBreaker.record_failure