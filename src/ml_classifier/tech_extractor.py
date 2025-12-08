#!/usr/bin/env python3
"""
Modern Hybrid Technical Information Extraction Pipeline (2024-2025)

Combines:
1. GLiNER - Zero-shot NER for custom technical entities
2. Semantic Pattern Classification - Embedding-based architecture pattern detection
3. Local LLM (Optional) - Deep structured extraction via Ollama

Replaces the outdated BERT NER approach with state-of-the-art methods.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer, util

# Ensure imports work
try:
    src_root = Path(__file__).parent.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from presentation.helpers.text_cleaner import clean_markdown
except Exception:
    def clean_markdown(text):
        return text or ""

# Cache directory
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# =============================================================================
# Data Models
# =============================================================================

class ExtractionSource(str, Enum):
    GLINER = "gliner"
    LLM = "llm"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


@dataclass
class TechEntity:
    """A technology/tool extracted from text"""
    name: str
    category: str  # language, framework, database, etc.
    confidence: float
    source: str = "gliner"
    context: Optional[str] = None


@dataclass
class ArchitecturePattern:
    """An architecture pattern detected in text"""
    name: str
    confidence: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class TechnicalExtraction:
    """Complete extraction result for a post"""
    post_id: str
    tech_stack: List[TechEntity] = field(default_factory=list)
    patterns: List[ArchitecturePattern] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    solutions: List[str] = field(default_factory=list)
    key_decisions: List[str] = field(default_factory=list)
    extraction_method: str = "hybrid"
    raw_llm_output: Optional[Dict[str, Any]] = None


# =============================================================================
# Device Detection
# =============================================================================

def get_device() -> Tuple[str, int]:
    """Auto-detect best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return 'cuda', 0
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps', -1
    return 'cpu', -1


# =============================================================================
# GLiNER Extractor - Zero-Shot NER
# =============================================================================

class GLiNERExtractor:
    """
    Zero-shot NER for technical entities using GLiNER.
    
    Unlike traditional BERT NER (Person, Org, Location, Misc),
    GLiNER can extract custom entity types without retraining.
    """
    
    # Custom labels for technical entity extraction
    TECH_LABELS = [
        "programming_language",
        "framework",
        "library",
        "database",
        "message_queue",
        "cloud_service",
        "container_technology",
        "monitoring_tool",
        "ci_cd_tool",
        "infrastructure_tool",
        "ml_framework",
        "data_processing_tool",
    ]
    
    # Company/publication names to exclude (not tech stack)
    COMPANY_BLOCKLIST = {
        "netflix", "airbnb", "uber", "lyft", "stripe", "spotify",
        "medium", "kickstarter", "tinder", "wise", "olx", "skyscanner",
        "facebook", "meta", "google", "amazon", "microsoft", "apple",
        "twitter", "linkedin", "github", "gitlab", "slack",
        "new york times", "nyt", "nytimes", "the new york times"
    }
    
    def __init__(self, model_name: str = "urchade/gliner_small-v2.1"):
        """
        Initialize GLiNER model.
        
        Args:
            model_name: GLiNER model to use. Options:
                - "urchade/gliner_small-v2.1" (~500MB, faster)
                - "urchade/gliner_medium-v2.1" (~1GB, balanced)
                - "urchade/gliner_large-v2.1" (~1.5GB, best quality)
        """
        self.model = None
        self.model_name = model_name
        self._loaded = False
        
    def _load(self):
        """Lazy load the model"""
        if self._loaded:
            return
            
        try:
            from gliner import GLiNER
            print(f"Loading GLiNER: {self.model_name}")
            self.model = GLiNER.from_pretrained(self.model_name)
            self._loaded = True
            print("✓ GLiNER loaded")
        except ImportError:
            print("⚠ GLiNER not installed. Run: pip install gliner")
            self._loaded = False
        except Exception as e:
            print(f"⚠ GLiNER load error: {e}")
            self._loaded = False
    
    def extract(self, text: str, threshold: float = 0.35) -> List[TechEntity]:
        """
        Extract technical entities from text.
        
        Args:
            text: Text to analyze
            threshold: Minimum confidence score (0-1)
            
        Returns:
            List of TechEntity objects
        """
        self._load()
        
        if not self._loaded or not self.model:
            return []
            
        # Clean and truncate text for model
        text_clean = clean_markdown(text)[:8000]  # GLiNER context limit
        
        if len(text_clean) < 50:
            return []
        
        try:
            entities = self.model.predict_entities(
                text_clean,
                self.TECH_LABELS,
                threshold=threshold
            )
            
            # Convert to TechEntity objects
            seen = set()
            results = []
            
            for e in entities:
                name = e["text"].strip()
                name_lower = name.lower()
                
                # Skip duplicates and very short names
                if name_lower in seen or len(name) < 2:
                    continue
                
                # Skip company/publication names (not tech stack)
                if name_lower in self.COMPANY_BLOCKLIST:
                    continue
                    
                seen.add(name_lower)
                results.append(TechEntity(
                    name=name,
                    category=e["label"],
                    confidence=round(e["score"], 3),
                    source="gliner"
                ))
            
            return sorted(results, key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            print(f"GLiNER extraction error: {e}")
            return []


# =============================================================================
# Architecture Pattern Classifier - Embedding-Based
# =============================================================================

class PatternClassifier:
    """
    Detect architecture patterns using semantic embeddings.
    
    Compares text segments against known pattern descriptions
    using cosine similarity.
    """
    
    # Architecture patterns with semantic descriptions for matching
    PATTERNS = {
        "Event Sourcing": [
            "event store", "event log", "replay events", "event stream",
            "append-only log", "event sourcing", "state from events",
            "immutable events", "event replay"
        ],
        "CQRS": [
            "command query responsibility segregation", "CQRS",
            "separate read write", "read model", "write model", 
            "projections", "query side", "command side"
        ],
        "Microservices": [
            "microservices", "micro services", "service mesh", "API gateway",
            "service discovery", "independent deployment", "bounded services",
            "decomposed services", "service-oriented"
        ],
        "Event-Driven Architecture": [
            "event-driven", "message queue", "pub/sub", "async messaging",
            "event bus", "choreography", "reactive", "publish subscribe",
            "message broker", "event notification"
        ],
        "Data Mesh": [
            "data mesh", "data products", "domain ownership",
            "federated governance", "self-serve data", "data as product",
            "domain-oriented data"
        ],
        "Domain-Driven Design": [
            "DDD", "bounded context", "aggregate", "domain model",
            "ubiquitous language", "domain events", "entity", "value object",
            "domain-driven design"
        ],
        "Saga Pattern": [
            "saga", "distributed transaction", "compensating transaction",
            "orchestration saga", "choreography saga", "long-running transaction"
        ],
        "Strangler Fig Pattern": [
            "strangler fig", "strangler pattern", "gradual migration",
            "facade pattern migration", "incremental rewrite", "legacy migration"
        ],
        "Circuit Breaker": [
            "circuit breaker", "fault tolerance", "bulkhead", "retry pattern",
            "graceful degradation", "failure isolation"
        ],
        "Service Mesh": [
            "service mesh", "sidecar proxy", "istio", "envoy", "linkerd",
            "traffic management", "service-to-service"
        ],
        "Stream Processing": [
            "stream processing", "real-time processing", "streaming pipeline",
            "event stream", "continuous processing", "streaming analytics"
        ],
        "Lambda Architecture": [
            "lambda architecture", "batch layer", "speed layer", "serving layer",
            "batch and streaming", "hybrid processing"
        ],
        "Kappa Architecture": [
            "kappa architecture", "streaming-first", "single processing layer",
            "stream-only"
        ],
        "Data Lake": [
            "data lake", "raw data storage", "schema on read", "unstructured data",
            "data reservoir"
        ],
        "Feature Store": [
            "feature store", "feature engineering", "ml features",
            "feature serving", "feature registry"
        ],
        "MLOps": [
            "mlops", "ml pipeline", "model deployment", "model serving",
            "model registry", "experiment tracking", "feature pipeline"
        ],
        "API Gateway Pattern": [
            "api gateway", "backend for frontend", "BFF", "api aggregation",
            "routing layer", "api management"
        ],
        "Sidecar Pattern": [
            "sidecar", "sidecar container", "ambassador pattern",
            "sidecar proxy", "attached container"
        ],
        "Change Data Capture": [
            "change data capture", "CDC", "database replication",
            "log-based replication", "debezium", "transaction log"
        ],
        "Polyglot Persistence": [
            "polyglot persistence", "multiple databases", "right tool for job",
            "database per service", "specialized storage"
        ]
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the pattern classifier with sentence transformer"""
        self.model = None
        self.model_name = model_name
        self.pattern_embeddings: Dict[str, torch.Tensor] = {}
        self._loaded = False
        
    def _load(self):
        """Lazy load model and build pattern embeddings"""
        if self._loaded:
            return
            
        device, _ = get_device()
        print(f"Loading Pattern Classifier: {self.model_name}")
        
        self.model = SentenceTransformer(
            self.model_name,
            cache_folder=str(CACHE_DIR),
            device=device
        )
        
        # Pre-compute pattern embeddings
        for pattern, keywords in self.PATTERNS.items():
            embeddings = self.model.encode(keywords, convert_to_tensor=True)
            self.pattern_embeddings[pattern] = embeddings.mean(dim=0)
        
        self._loaded = True
        print("✓ Pattern Classifier loaded")
    
    def classify(self, text: str, threshold: float = 0.40) -> List[ArchitecturePattern]:
        """
        Detect architecture patterns in text.
        
        Args:
            text: Text to analyze
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of detected ArchitecturePattern objects
        """
        self._load()
        
        if not self._loaded or not self.model:
            return []
        
        # Split into sentences for evidence extraction
        text_clean = clean_markdown(text)
        sentences = [
            s.strip() for s in text_clean.replace('\n', ' ').split('.')
            if len(s.strip()) > 30
        ]
        
        if not sentences:
            return []
        
        try:
            # Encode all sentences
            sentence_embeddings = self.model.encode(
                sentences[:100],  # Limit for performance
                convert_to_tensor=True
            )
            
            detected = []
            
            for pattern, pattern_emb in self.pattern_embeddings.items():
                # Compute similarity with all sentences
                similarities = util.cos_sim(sentence_embeddings, pattern_emb).squeeze()
                
                # Handle single sentence case
                if len(similarities.shape) == 0:
                    similarities = similarities.unsqueeze(0)
                
                # Get top matching sentences
                top_indices = similarities.argsort(descending=True)[:3]
                top_scores = [similarities[i].item() for i in top_indices]
                
                if top_scores[0] > threshold:
                    evidence = [
                        sentences[i][:200] for i in top_indices
                        if similarities[i] > threshold * 0.8
                    ]
                    detected.append(ArchitecturePattern(
                        name=pattern,
                        confidence=round(top_scores[0], 3),
                        evidence=evidence[:2]
                    ))
            
            return sorted(detected, key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            print(f"Pattern classification error: {e}")
            return []


# =============================================================================
# LLM Structured Extractor - Deep Extraction via Ollama
# =============================================================================

class LLMStructuredExtractor:
    """
    Deep extraction using local LLM (Ollama) with structured JSON output.
    
    Uses models like Qwen2.5-7B or Phi-3 for:
    - Problem/solution extraction
    - Key architectural decisions
    - Verification of NER results
    """
    
    SYSTEM_PROMPT = """You are a technical architecture analyst specializing in 
extracting structured information from engineering blog posts.

Be precise and factual:
- Only include technologies explicitly mentioned and actually used
- Distinguish between technologies used vs. mentioned for comparison
- Identify actual architectural decisions made, not general discussions
- Extract specific problems faced and their concrete solutions
- Focus on infrastructure, data systems, and software architecture"""

    EXTRACTION_PROMPT = """Analyze this engineering blog post and extract technical details.

{hints_section}

Blog Post:
{text}

Extract as JSON with these exact fields:
{{
    "tech_stack": {{
        "languages": ["programming languages used"],
        "frameworks": ["frameworks and libraries"],
        "databases": ["databases and data stores"],
        "infrastructure": ["cloud services, containers, orchestration"],
        "messaging": ["queues, streaming platforms"],
        "observability": ["monitoring, logging, tracing"]
    }},
    "architecture_patterns": [
        {{"name": "pattern name", "evidence": "brief quote from text"}}
    ],
    "problems_addressed": ["specific technical challenges faced"],
    "solutions_implemented": ["how problems were solved"],
    "key_decisions": ["architectural decisions with brief rationale"]
}}

Return ONLY valid JSON, no markdown or explanation."""

    def __init__(self, model: str = "qwen2.5:14b"):
        """
        Initialize LLM extractor.
        
        Args:
            model: Ollama model name. Recommended:
                - "qwen2.5:14b" - Best JSON reliability
                - "phi3:medium" - Good alternative
                - "llama3.1:8b" - Faster but less reliable JSON
        """
        self.model = model
        self._available = None
        
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        if self._available is not None:
            return self._available
            
        try:
            import ollama
            ollama.list()
            self._available = True
            print(f"✓ Ollama available, using model: {self.model}")
        except Exception as e:
            print(f"⚠ Ollama not available: {e}")
            self._available = False
            
        return self._available
    
    def extract(
        self,
        text: str,
        gliner_hints: Optional[List[TechEntity]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract structured technical information using LLM.
        
        Args:
            text: Text to analyze
            gliner_hints: Optional tech entities from GLiNER to verify/expand
            
        Returns:
            Dictionary with extracted information, or None on failure
        """
        if not self._check_ollama():
            return None
            
        import ollama
        
        # Build hints section
        hints_section = ""
        if gliner_hints:
            hints = ", ".join(set(e.name for e in gliner_hints[:15]))
            hints_section = f"GLiNER detected these technologies (verify and expand): {hints}\n"
        
        # Truncate text for context window
        text_clean = clean_markdown(text)[:6000]
        
        prompt = self.EXTRACTION_PROMPT.format(
            hints_section=hints_section,
            text=text_clean
        )
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                format="json",
                options={
                    "temperature": 0.1,
                    "num_ctx": 8192
                }
            )
            
            content = response['message']['content']
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"LLM JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return None


# =============================================================================
# Main Pipeline
# =============================================================================

class TechExtractionPipeline:
    """
    Hybrid technical extraction pipeline.
    
    Stages:
    1. GLiNER - Fast zero-shot NER for tech entities (~0.1s/post)
    2. Pattern Classifier - Embedding-based architecture detection (~0.05s/post)
    3. LLM (Optional) - Deep structured extraction (~3-5s/post)
    
    Performance estimate for 300 posts:
    - GLiNER only: ~30 seconds
    - With patterns: ~45 seconds
    - Full hybrid with LLM: ~15-25 minutes
    """
    
    def __init__(
        self,
        use_gliner: bool = True,
        use_patterns: bool = True,
        use_llm: bool = False,
        gliner_model: str = "urchade/gliner_small-v2.1",
        llm_model: str = "qwen2.5:14b"
    ):
        """
        Initialize the extraction pipeline.
        
        Args:
            use_gliner: Enable GLiNER tech extraction
            use_patterns: Enable pattern classification
            use_llm: Enable LLM deep extraction (slower but more detailed)
            gliner_model: GLiNER model to use
            llm_model: Ollama model for LLM extraction
        """
        self.use_gliner = use_gliner
        self.use_patterns = use_patterns
        self.use_llm = use_llm
        
        # Lazy initialization
        self._gliner: Optional[GLiNERExtractor] = None
        self._patterns: Optional[PatternClassifier] = None
        self._llm: Optional[LLMStructuredExtractor] = None
        
        self.gliner_model = gliner_model
        self.llm_model = llm_model
        
    def _get_gliner(self) -> GLiNERExtractor:
        if self._gliner is None:
            self._gliner = GLiNERExtractor(self.gliner_model)
        return self._gliner
    
    def _get_patterns(self) -> PatternClassifier:
        if self._patterns is None:
            self._patterns = PatternClassifier()
        return self._patterns
    
    def _get_llm(self) -> LLMStructuredExtractor:
        if self._llm is None:
            self._llm = LLMStructuredExtractor(self.llm_model)
        return self._llm
    
    def process(self, post_id: str, text: str) -> TechnicalExtraction:
        """
        Process a single post and extract technical information.
        
        Args:
            post_id: Unique identifier for the post
            text: Post content (markdown or plain text)
            
        Returns:
            TechnicalExtraction with all extracted data
        """
        tech_stack: List[TechEntity] = []
        patterns: List[ArchitecturePattern] = []
        problems: List[str] = []
        solutions: List[str] = []
        decisions: List[str] = []
        llm_output = None
        
        # Stage 1: GLiNER extraction
        if self.use_gliner:
            tech_stack = self._get_gliner().extract(text)
        
        # Stage 2: Pattern classification
        if self.use_patterns:
            patterns = self._get_patterns().classify(text)
        
        # Stage 3: LLM deep extraction (optional)
        if self.use_llm and len(text) > 500:
            llm_output = self._get_llm().extract(text, tech_stack)
            
            if llm_output:
                # Merge LLM results
                tech_stack, patterns = self._merge_llm_results(
                    tech_stack, patterns, llm_output
                )
                problems = llm_output.get("problems_addressed", [])
                solutions = llm_output.get("solutions_implemented", [])
                decisions = llm_output.get("key_decisions", [])
        
        # Determine extraction method used
        method = []
        if self.use_gliner and tech_stack:
            method.append("gliner")
        if self.use_patterns and patterns:
            method.append("patterns")
        if self.use_llm and llm_output:
            method.append("llm")
        
        return TechnicalExtraction(
            post_id=post_id,
            tech_stack=tech_stack,
            patterns=patterns,
            problems=problems,
            solutions=solutions,
            key_decisions=decisions,
            extraction_method="+".join(method) if method else "none",
            raw_llm_output=llm_output
        )
    
    def _merge_llm_results(
        self,
        tech_stack: List[TechEntity],
        patterns: List[ArchitecturePattern],
        llm_output: Dict[str, Any]
    ) -> Tuple[List[TechEntity], List[ArchitecturePattern]]:
        """Merge LLM results with GLiNER/pattern results"""
        
        # Track existing entities by lowercase name
        existing_tech = {e.name.lower() for e in tech_stack}
        
        # Add LLM-verified tech entities
        llm_tech = llm_output.get("tech_stack", {})
        category_map = {
            "languages": "programming_language",
            "frameworks": "framework",
            "databases": "database",
            "infrastructure": "infrastructure_tool",
            "messaging": "message_queue",
            "observability": "monitoring_tool"
        }
        
        for category, items in llm_tech.items():
            cat_label = category_map.get(category, category)
            for item in (items or []):
                if isinstance(item, str) and item.lower() not in existing_tech:
                    tech_stack.append(TechEntity(
                        name=item,
                        category=cat_label,
                        confidence=0.75,
                        source="llm"
                    ))
                    existing_tech.add(item.lower())
        
        # Boost confidence for entities found by both
        for entity in tech_stack:
            if entity.source == "gliner":
                # Check if LLM also found it
                for items in llm_tech.values():
                    if items and entity.name.lower() in [i.lower() for i in items if isinstance(i, str)]:
                        entity.confidence = min(0.95, entity.confidence + 0.15)
                        entity.source = "hybrid"
                        break
        
        # Add LLM patterns
        existing_patterns = {p.name.lower() for p in patterns}
        for p in llm_output.get("architecture_patterns", []):
            if isinstance(p, dict):
                name = p.get("name", "")
                if name and name.lower() not in existing_patterns:
                    patterns.append(ArchitecturePattern(
                        name=name,
                        confidence=0.70,
                        evidence=[p.get("evidence", "")]
                    ))
        
        return tech_stack, patterns
    
    def process_batch(
        self,
        posts: List[Dict[str, str]],
        progress_callback=None
    ) -> List[TechnicalExtraction]:
        """
        Process multiple posts.
        
        Args:
            posts: List of {"id": ..., "content": ...} dicts
            progress_callback: Optional callback(current, total, post_id)
            
        Returns:
            List of TechnicalExtraction results
        """
        results = []
        
        for i, post in enumerate(posts):
            result = self.process(post["id"], post["content"])
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(posts), post["id"])
        
        return results


# =============================================================================
# Utility Functions
# =============================================================================

def extraction_to_dict(extraction: TechnicalExtraction) -> Dict[str, Any]:
    """Convert TechnicalExtraction to a dictionary for JSON serialization"""
    return {
        "post_id": extraction.post_id,
        "tech_stack": [
            {
                "name": e.name,
                "category": e.category,
                "confidence": e.confidence,
                "source": e.source
            }
            for e in extraction.tech_stack
        ],
        "patterns": [
            {
                "name": p.name,
                "confidence": p.confidence,
                "evidence": p.evidence
            }
            for p in extraction.patterns
        ],
        "problems": extraction.problems,
        "solutions": extraction.solutions,
        "key_decisions": extraction.key_decisions,
        "extraction_method": extraction.extraction_method
    }


def dict_to_extraction(data: Dict[str, Any]) -> TechnicalExtraction:
    """Convert dictionary back to TechnicalExtraction"""
    return TechnicalExtraction(
        post_id=data.get("post_id", ""),
        tech_stack=[
            TechEntity(
                name=t["name"],
                category=t.get("category", "unknown"),
                confidence=t.get("confidence", 0.5),
                source=t.get("source", "unknown")
            )
            for t in data.get("tech_stack", [])
        ],
        patterns=[
            ArchitecturePattern(
                name=p["name"],
                confidence=p.get("confidence", 0.5),
                evidence=p.get("evidence", [])
            )
            for p in data.get("patterns", [])
        ],
        problems=data.get("problems", []),
        solutions=data.get("solutions", []),
        key_decisions=data.get("key_decisions", []),
        extraction_method=data.get("extraction_method", "unknown")
    )


# =============================================================================
# CLI / Testing
# =============================================================================

if __name__ == "__main__":
    # Test the pipeline
    sample_text = """
    How We Migrated to Event Sourcing at Scale
    
    Our team faced a critical challenge: our monolithic PostgreSQL database 
    couldn't handle the write throughput we needed. After evaluating several 
    approaches, we decided to implement event sourcing using Apache Kafka.
    
    The migration strategy used the strangler fig pattern - we gradually 
    moved functionality from our Django backend to new Go microservices. 
    Each service owns its Kafka topics and maintains its own read models 
    in Elasticsearch for fast queries.
    
    Key decisions:
    - Chose Kafka over RabbitMQ for durability and replay capability
    - Implemented CQRS to separate our write and read paths
    - Used Kubernetes for orchestration with Istio service mesh
    - Added distributed tracing with Jaeger
    
    The results: 10x improvement in write throughput and much better 
    audit capabilities through our event log.
    """
    
    print("=" * 60)
    print("Testing Tech Extraction Pipeline")
    print("=" * 60)
    
    # Initialize pipeline (no LLM for quick test)
    pipeline = TechExtractionPipeline(
        use_gliner=True,
        use_patterns=True,
        use_llm=False  # Set to True if Ollama is running
    )
    
    # Process sample
    result = pipeline.process("sample-001", sample_text)
    
    print(f"\n📦 Tech Stack ({len(result.tech_stack)} items):")
    for tech in result.tech_stack[:10]:
        print(f"  • {tech.name} ({tech.category}) "
              f"[conf: {tech.confidence:.2f}, src: {tech.source}]")
    
    print(f"\n🏗️ Architecture Patterns ({len(result.patterns)}):")
    for pattern in result.patterns[:5]:
        print(f"  • {pattern.name} [conf: {pattern.confidence:.2f}]")
        if pattern.evidence:
            print(f"    Evidence: \"{pattern.evidence[0][:80]}...\"")
    
    print(f"\n📊 Extraction Method: {result.extraction_method}")
    
    # Export as JSON
    print("\n📄 JSON Output:")
    print(json.dumps(extraction_to_dict(result), indent=2))
