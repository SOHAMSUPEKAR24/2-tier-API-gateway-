"""
tier2/
──────
Deep Verification Model Module for Two-Tier API Gateway Architecture.
Renamed from 'sandbox' per Project Rules:
  Only true isolated execution environments (e.g. CAPE Sandbox) are called sandboxes.
  Stateless, heavy machine learning verifiers are called Deep Verification Models.
"""

from .track_a_http import TrackAHTTPVerifier, extract_track_a_fast_features, extract_raw_http_text
from .track_b_pe import TrackBPEVerifier, extract_track_b_tier1_features, extract_track_b_tier2_features

__all__ = [
    "TrackAHTTPVerifier",
    "extract_track_a_fast_features",
    "extract_raw_http_text",
    "TrackBPEVerifier",
    "extract_track_b_tier1_features",
    "extract_track_b_tier2_features",
]
