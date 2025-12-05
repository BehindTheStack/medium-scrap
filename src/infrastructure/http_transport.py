"""HTTP transport abstraction to make adapters testable and pluggable.

Provides a simple interface with `post` and `get` methods. The default
implementation uses curl-cffi which impersonates Chrome's TLS fingerprint
to avoid anti-bot detection.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from curl_cffi import requests as curl_requests


class HTTPTransport:
    """Abstract transport interface. Concrete transports should implement
    `post` and `get` with compatible signatures returning Response-like objects.
    """

    def post(self, url: str, headers: Dict[str, str], json: Dict[str, Any], timeout: float = 30.0):
        raise NotImplementedError()

    def get(self, url: str, headers: Dict[str, str], follow_redirects: bool = True, timeout: float = 30.0):
        raise NotImplementedError()


class HttpxTransport(HTTPTransport):
    """curl-cffi based transport that impersonates Chrome to bypass anti-bot detection."""

    def _filter_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Remove headers that conflict with impersonation (curl-cffi sets these automatically)."""
        skip_keys = {'user-agent', 'accept-encoding', 'accept-language'}
        return {k: v for k, v in headers.items() if k.lower() not in skip_keys}

    def post(self, url: str, headers: Dict[str, str], json: Dict[str, Any], timeout: float = 30.0):
        return curl_requests.post(
            url,
            headers=self._filter_headers(headers),
            json=json,
            impersonate="chrome",
            verify=False,
            timeout=timeout
        )

    def get(self, url: str, headers: Dict[str, str], follow_redirects: bool = True, timeout: float = 30.0):
        return curl_requests.get(
            url,
            headers=self._filter_headers(headers),
            impersonate="chrome",
            verify=False,
            timeout=timeout,
            allow_redirects=follow_redirects
        )


__all__ = ["HTTPTransport", "HttpxTransport"]
