"""Tests for AzureFoundryClient._extract_usage.

We don't import the openai SDK; just feed the static method response
stand-ins shaped like the real SDK response (object-style and
dict-style).
"""

from __future__ import annotations

import unittest

from src.ai.azure_client import AzureFoundryClient


class _Obj:
    """Tiny attribute-bag stand-in for SDK objects."""

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class ExtractUsageTests(unittest.TestCase):
    def test_object_style_with_cached_object(self) -> None:
        resp = _Obj(
            usage=_Obj(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                prompt_tokens_details=_Obj(cached_tokens=20),
            )
        )
        p, c, cached, total = AzureFoundryClient._extract_usage(resp)
        self.assertEqual((p, c, cached, total), (100, 50, 20, 150))

    def test_object_style_without_cached(self) -> None:
        resp = _Obj(
            usage=_Obj(prompt_tokens=80, completion_tokens=40, total_tokens=120)
        )
        p, c, cached, total = AzureFoundryClient._extract_usage(resp)
        self.assertEqual((p, c, cached, total), (80, 40, 0, 120))

    def test_dict_style_with_cached(self) -> None:
        resp = _Obj(usage={
            "prompt_tokens": 70,
            "completion_tokens": 30,
            "total_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 10},
        })
        p, c, cached, total = AzureFoundryClient._extract_usage(resp)
        self.assertEqual((p, c, cached, total), (70, 30, 10, 100))

    def test_dict_style_missing_total_inferred(self) -> None:
        resp = _Obj(usage={"prompt_tokens": 10, "completion_tokens": 5})
        p, c, cached, total = AzureFoundryClient._extract_usage(resp)
        self.assertEqual((p, c, cached, total), (10, 5, 0, 15))

    def test_missing_usage_returns_zeros(self) -> None:
        self.assertEqual(
            AzureFoundryClient._extract_usage(_Obj(usage=None)),
            (0, 0, 0, 0),
        )

    def test_response_without_usage_attribute(self) -> None:
        self.assertEqual(
            AzureFoundryClient._extract_usage(_Obj()),
            (0, 0, 0, 0),
        )

    def test_object_style_details_as_dict(self) -> None:
        resp = _Obj(
            usage=_Obj(
                prompt_tokens=50, completion_tokens=20, total_tokens=70,
                prompt_tokens_details={"cached_tokens": 15},
            )
        )
        p, c, cached, total = AzureFoundryClient._extract_usage(resp)
        self.assertEqual((p, c, cached, total), (50, 20, 15, 70))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
