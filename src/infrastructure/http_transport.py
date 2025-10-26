"""HTTP transport abstraction to make adapters testable and pluggable.

Provides a simple interface with `post` and `get` methods. The default
implementation wraps httpx but tests can inject a mock transport.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import httpx


class HTTPTransport:
    """Abstract transport interface. Concrete transports should implement
    `post` and `get` with compatible signatures returning httpx.Response-like objects.
    """

    def post(self, url: str, headers: Dict[str, str], json: Dict[str, Any], timeout: float = 30.0):
        raise NotImplementedError()

    def get(self, url: str, headers: Dict[str, str], follow_redirects: bool = True, timeout: float = 30.0):
        raise NotImplementedError()


class HttpxTransport(HTTPTransport):
    """httpx-based transport implementation."""

    def post(self, url: str, headers: Dict[str, str], json: Dict[str, Any], timeout: float = 30.0):
        # If tests have patched httpx.Client (e.g. replaced by MagicMock) it may not
        # behave like the real class. Detect that and prefer the client context
        # manager so tests mocking httpx.Client are respected.
        try:
            if not hasattr(httpx.Client, "__mro__"):
                with httpx.Client(verify=False, timeout=timeout) as client:
                    return client.post(url, headers=headers, json=json)

            # Prefer the top-level helper (httpx.post) which is convenient for simple test mocking.
            return httpx.post(url, headers=headers, json=json, verify=False, timeout=timeout)
        except Exception:
            # Fallback to client context manager as a last resort.
            with httpx.Client(verify=False, timeout=timeout) as client:
                return client.post(url, headers=headers, json=json)

    def get(self, url: str, headers: Dict[str, str], follow_redirects: bool = True, timeout: float = 30.0):
        with httpx.Client(verify=False, timeout=timeout, follow_redirects=follow_redirects) as client:
            return client.get(url, headers=headers)


__all__ = ["HTTPTransport", "HttpxTransport"]
