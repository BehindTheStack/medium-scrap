"""
ML Discovery Data Schemas
"""

from typing import List, Optional, TypedDict


class TechStackItem(TypedDict):
    """Technology stack item with score"""
    name: str
    score: float


class PatternItem(TypedDict):
    """Pattern item with confidence"""
    pattern: str
    confidence: float


class MLDiscoveryData(TypedDict):
    """Complete ML discovery data structure"""
    layers: List[str]
    tech_stack: List[TechStackItem]
    patterns: List[PatternItem]
    solutions: List[str]
    problem: Optional[str]
    approach: Optional[str]
