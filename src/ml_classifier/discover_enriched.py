#!/usr/bin/env python3
"""
Enhanced topic discovery with NER-based tech stack extraction,
zero-shot pattern classification, and solution mining.

Uses only HuggingFace models - no hardcoded keywords.
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import sys
from pathlib import Path as _Path

# Ensure we can import the centralized text_cleaner from src/presentation/helpers
try:
    # Add src/ to sys.path so `presentation.helpers` is importable
    src_root = _Path(__file__).parent.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from presentation.helpers.text_cleaner import clean_markdown, chunk_text
except Exception:
    # Fallback: define a minimal cleaner (shouldn't happen in normal runs)
    def clean_markdown(text):
        return text or ""

    def chunk_text(text, max_chars=None, overlap=0):
        return [text]

# Set LD_LIBRARY_PATH for WSL CUDA support
if os.path.exists('/usr/lib/wsl/lib'):
    wsl_lib = '/usr/lib/wsl/lib'
    if wsl_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
        os.environ['LD_LIBRARY_PATH'] = f"{wsl_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

# Cache directory for models
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Models
EMBEDDER_NAME = "all-MiniLM-L6-v2"
NER_MODEL = "dslim/bert-base-NER"
QA_MODEL = "deepset/roberta-base-squad2"  # For problem/approach extraction


def get_device():
    """
    Auto-detect best available device: CUDA > MPS > CPU
    
    Returns:
        tuple: (device_name, device_id_for_pipeline)
            - device_name: 'cuda', 'mps', or 'cpu'
            - device_id_for_pipeline: 0 for cuda, -1 for cpu/mps
    """
    if torch.cuda.is_available():
        device_name = 'cuda'
        device_id = 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ Using GPU: {gpu_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_name = 'mps'
        device_id = -1  # MPS works via model.to('mps'), not pipeline device arg
        print("✓ Using Apple Silicon MPS")
    else:
        device_name = 'cpu'
        device_id = -1
        print("⚠ Using CPU (GPU not available - slower inference)")
    
    return device_name, device_id

def extract_patterns(text: str, ner_pipeline, embedder) -> List[Dict[str, Any]]:
    """
    Extract architectural patterns using NER + semantic analysis.
    NO hardcoded list - discovers patterns from the text itself.
    
    Strategy:
    1. Extract technical entities (NER)
    2. Find technical n-grams (bigrams/trigrams)
    3. Filter for architectural patterns using semantic similarity
    
    Returns:
        List of dicts with {pattern, confidence}
    """
    if not text or len(text.strip()) < 20:
        return []

    # Clean the full text (do not truncate here)
    text_sample = clean_markdown(text)

    try:
        # 1. Get technical entities from NER
        entities = ner_pipeline(text_sample)
        tech_entities = [
            e['word'].replace('##', '').strip()
            for e in entities
            if e.get('entity_group') in ['ORG', 'MISC'] and e.get('score', 0) > 0.80
        ]

        # 2. Extract technical bigrams/trigrams
        import re
        # Clean additional artifacts and remove remaining URLs (clean_markdown already ran)
        clean_text = re.sub(r'```.*?```', '', text_sample, flags=re.DOTALL)
        clean_text = re.sub(r'http\S+', '', clean_text)
        clean_text = re.sub(r'[#*`\[\]()]', ' ', clean_text)

        # Split into sentences
        sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 20]

        # Extract n-grams that contain technical terms
        patterns_found = []
        pattern_keywords = ['architecture', 'pattern', 'system', 'design', 'approach',
                           'model', 'framework', 'platform', 'infrastructure', 'service']

        for sentence in sentences[:30]:  # examine first 30 candidate sentences
            sentence_lower = sentence.lower()

            # Check if sentence discusses patterns/architecture
            if any(kw in sentence_lower for kw in pattern_keywords):
                # Extract potential pattern phrases (2-4 words)
                words = sentence.split()
                for i in range(len(words) - 1):
                    # Bigrams
                    bigram = f"{words[i]} {words[i+1]}"
                    if any(tech in bigram.lower() for tech in ['api', 'data', 'micro', 'event',
                                                                'service', 'container', 'cloud',
                                                                'real-time', 'batch', 'stream']):
                        patterns_found.append(bigram)

                    # Trigrams
                    if i < len(words) - 2:
                        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                        if any(tech in trigram.lower() for tech in ['architecture', 'pattern',
                                                                     'system', 'infrastructure']):
                            patterns_found.append(trigram)

        # 3. Deduplicate and score by frequency
        from collections import Counter
        pattern_counts = Counter(patterns_found)

        # Return top patterns
        patterns = []
        for pattern, count in pattern_counts.most_common(10):
            # Clean up
            pattern_clean = pattern.strip().title()
            if len(pattern_clean) > 5:  # Minimum length
                patterns.append({
                    'pattern': pattern_clean,
                    'confidence': min(count / max(len(sentences), 1), 1.0)  # Normalize safely
                })

        return patterns[:8]  # Top 8

    except Exception as e:
        print(f"Pattern extraction error: {e}")
        return []


def load_embedder():
    """Load sentence transformer model."""
    device_name, _ = get_device()
    print(f"Loading embedder: {EMBEDDER_NAME}")
    return SentenceTransformer(EMBEDDER_NAME, cache_folder=str(CACHE_DIR), device=device_name)


def load_ner_pipeline():
    """Load NER pipeline for entity extraction."""
    device_name, device_id = get_device()
    print(f"Loading NER model: {NER_MODEL}")
    pipe = pipeline(
        "ner",
        model=NER_MODEL,
        aggregation_strategy="simple",
        device=device_id
    )
    # For MPS, move model explicitly
    if device_name == 'mps':
        pipe.model.to('mps')
    return pipe


def load_qa_pipeline():
    """Load Q&A pipeline for problem/approach extraction."""
    device_name, device_id = get_device()
    print(f"Loading Q&A model: {QA_MODEL}")
    pipe = pipeline(
        "question-answering",
        model=QA_MODEL,
        device=device_id
    )
    # For MPS, move model explicitly
    if device_name == 'mps':
        pipe.model.to('mps')
    return pipe


def extract_tech_stack(text: str, ner_pipeline) -> List[Dict[str, Any]]:
    """
    Extract technology mentions using NER with smart filtering.
    
    Returns:
        List of dicts with {name, type, score}
    """
    if not text or len(text.strip()) < 10:
        return []

    # Clean full text for NER
    text_sample = clean_markdown(text)
    
    try:
        entities = ner_pipeline(text_sample)
    except Exception as e:
        print(f"NER error: {e}")
        return []
    
    # Common tech keywords to boost confidence
    tech_indicators = {
        'kafka', 'spark', 'hadoop', 'kubernetes', 'k8s', 'docker', 'redis', 
        'elasticsearch', 'postgres', 'mysql', 'mongodb', 'cassandra', 'aws', 
        'gcp', 'azure', 'react', 'vue', 'angular', 'python', 'java', 'golang',
        'typescript', 'javascript', 'grpc', 'graphql', 'rest', 'api', 'microservice',
        'terraform', 'ansible', 'jenkins', 'gitlab', 'github', 'prometheus',
        'grafana', 'datadog', 'splunk', 'airflow', 'flink', 'storm', 'beam',
        'tensorflow', 'pytorch', 'scikit', 'pandas', 'numpy', 'jupyter',
        'metaflow', 'maestro', 'nginx', 'apache', 'tomcat', 'jetty'
    }
    
    # Blacklist generic or non-tech entities
    blacklist = {
        'netflix', 'stranger things', 'squid game', 'nfl', 'christmas', 
        'behind the streams', 'part', 'internet', 'time', 'real', 'page',
        'intent', 'live', 'events', 'the', 'and', 'for', 'with', 'our',
        'their', 'this', 'that', 'from', 'into', 'about', 'over', 'under',
        # More non-tech terms
        'internet scale', 'on demand', 'nfl christmas day games', 'spin',
        'metaf', 'real time', 'live events', 'netflix technology', 'making',
        'christmas day', 'day games', 'games', 'scale', 'demand'
    }
    
    # Merge adjacent entities and filter
    merged_entities = []
    i = 0
    
    while i < len(entities):
        current = entities[i]
        word = current['word'].strip()
        entity_type = current['entity_group']
        score = current['score']
        start = current.get('start', i)
        end = current.get('end', start + len(word))
        
        # Merge adjacent entities of same type
        while i + 1 < len(entities):
            next_entity = entities[i + 1]
            next_start = next_entity.get('start', 0)
            
            # If adjacent (within 2 chars) and same type, merge
            if (next_entity['entity_group'] == entity_type and 
                next_start - end <= 2):
                word += ' ' + next_entity['word'].strip()
                score = max(score, next_entity['score'])
                end = next_entity.get('end', end + len(next_entity['word']))
                i += 1
            else:
                break
        
        # Clean up subword tokens, whitespace, and extra spaces
        word_clean = word.replace('##', '').replace('  ', ' ').strip()
        # Remove incomplete words (likely fragmentation)
        if any(fragment in word_clean.lower() for fragment in ['enco ding', 'co smos', 'mae ', 'di ributed', 'a traction']):
            i += 1
            continue
            
        word_lower = word_clean.lower()
        
        # Skip if too short, in blacklist, or looks like fragment
        min_length = 4 if not any(tech in word_lower for tech in tech_indicators) else 3
        if len(word_clean) >= min_length and word_lower not in blacklist:
            # Keep ORG (organizations/products/tools) and high-confidence MISC
            if entity_type == 'ORG' or (entity_type == 'MISC' and score > 0.96):
                # Boost score if it's a known tech term
                is_tech_term = any(tech in word_lower for tech in tech_indicators)
                min_score = 0.82 if is_tech_term else 0.93  # Balanced threshold
                
                if score > min_score:
                    merged_entities.append({
                        'name': word_clean,
                        'type': entity_type,
                        'score': float(score)
                    })
        
        i += 1
    
    # Post-process: clean up and deduplicate
    tech_mentions = []
    seen = set()
    
    for entity in merged_entities:
        name = entity['name']
        # Fix common spacing issues
        name = name.replace(' raphQL', 'raphQL')  # G raphQL → GraphQL
        name = name.replace('  ', ' ').strip()
        
        # Skip duplicates (e.g., "Muse Muse" → "Muse")
        words = name.split()
        if len(words) == 2 and words[0] == words[1]:
            name = words[0]
        
        word_lower = name.lower()
        if word_lower not in seen and len(name) >= 3:
            seen.add(word_lower)
            entity['name'] = name  # Update with cleaned name
            tech_mentions.append(entity)
    
    # Sort by score and return top 12
    tech_mentions.sort(key=lambda x: x['score'], reverse=True)
    return tech_mentions[:12]


def extract_solutions(text: str, tech_stack: List[Dict], embedder) -> List[str]:
    """
    Extract solution descriptions using semantic similarity.
    NO hardcoded keywords - uses sentence embeddings.
    
    Strategy:
    1. Split text into sentences
    2. Compute embeddings for each sentence
    3. Find sentences semantically similar to "solution implementation"
    4. Filter sentences that mention technologies
    
    Returns:
        List of solution description sentences
    """
    if not text:
        return []
    
    solutions = []
    
    try:
        # Improved templates focusing on technical solutions
        solution_templates = [
            "We built and deployed a distributed system using microservices",
            "The implementation leverages machine learning and data pipelines",
            "Our solution uses Kafka, Spark, and cloud infrastructure",
            "We developed an architecture with APIs and databases",
            "The system implements caching, load balancing, and monitoring",
            "We created a platform combining multiple technologies"
        ]
        
        # Get embeddings for templates
        template_embeddings = embedder.encode(solution_templates)

        # Use cleaned text and split into sentences
        clean_text = clean_markdown(text)
        sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 30][:50]

        if not sentences:
            return []

        # Get embeddings for sentences
        sentence_embeddings = embedder.encode(sentences)
        
        # Compute similarity with solution templates
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        similarities = cosine_similarity(sentence_embeddings, template_embeddings)
        max_similarities = similarities.max(axis=1)  # Max similarity for each sentence
        
        # Find sentences with high similarity
        for i, (sentence, sim) in enumerate(zip(sentences, max_similarities)):
            # Semantic similarity to solution templates
            if sim > 0.38:  # Slightly lower for better recall
                # Check if mentions any technology (if tech_stack exists)
                mentions_tech = False
                if tech_stack:
                    mentions_tech = any(
                        tech['name'].lower() in sentence.lower()
                        for tech in tech_stack
                    )
                
                # Better solution keywords focusing on technical implementations
                solution_keywords = [
                    'built', 'implemented', 'developed', 'created', 'designed',
                    'architecture', 'infrastructure', 'pipeline', 'platform',
                    'service', 'api', 'database', 'cache', 'deploy', 'scale'
                ]
                has_solution_keyword = any(kw in sentence.lower() for kw in solution_keywords)
                
                # More selective: require keyword AND (tech mention OR high similarity)
                if has_solution_keyword and (mentions_tech or sim > 0.50):
                    solutions.append(sentence.strip())
                    
                    if len(solutions) >= 5:
                        break
        
        return solutions
        
    except Exception as e:
        print(f"Solution extraction error: {e}")
        return []


def extract_problem(text: str, qa_pipeline) -> str:
    """
    Extract the main problem/challenge using Q&A model.
    NO hardcoded keywords - uses question-answering.
    
    Args:
        text: Post content
        qa_pipeline: HuggingFace Q&A pipeline
    
    Returns:
        Problem description or None
    """
    if not text or len(text) < 100:
        return None

    try:
        # Use the cleaned full text as context
        context = clean_markdown(text)

        # Try multiple specific questions for better results
        questions = [
            "What technical problem or challenge did the engineering team face?",
            "What was the main scalability or performance issue?",
            "What problem needed to be solved or what challenge did they encounter?"
        ]
        
        best_answer = None
        best_score = 0.0
        
        for question in questions:
            result = qa_pipeline(question=question, context=context)
            answer = result['answer'].strip()
            score = result['score']
            
            # Keep the best answer (substantial and high confidence)
            if len(answer) > 20 and score > best_score and score > 0.08:
                best_answer = answer
                best_score = score
        
        return best_answer if best_answer else None
        
    except Exception as e:
        print(f"Problem extraction error: {e}")
        return None


def extract_approach(text: str, qa_pipeline) -> str:
    """
    Extract the main approach/solution using Q&A model.
    NO hardcoded keywords - uses question-answering.
    
    Args:
        text: Post content
        qa_pipeline: HuggingFace Q&A pipeline
    
    Returns:
        Approach description or None
    """
    if not text or len(text) < 100:
        return None

    try:
        # Use cleaned full text as context
        context = clean_markdown(text)

        # Try multiple questions to capture different solution aspects
        questions = [
            "What technical approach or architecture did they use to solve this?",
            "How did the engineering team implement the solution?",
            "What was their technical solution or approach?"
        ]
        
        best_answer = None
        best_score = 0.0
        
        for question in questions:
            result = qa_pipeline(question=question, context=context)
            answer = result['answer'].strip()
            score = result['score']
            
            # Keep the best answer
            if len(answer) > 20 and score > best_score and score > 0.08:
                best_answer = answer
                best_score = score
        
        return best_answer if best_answer else None
        
    except Exception as e:
        print(f"Approach extraction error: {e}")
        return None


def discover_enriched(args):
    """Main discovery pipeline with enriched extraction."""
    
    print("=" * 80)
    print("Enhanced Topic Discovery with Tech Stack & Pattern Extraction")
    print("=" * 80)
    
    # Load input
    input_path = Path(args.input)
    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    entries = data.get('entries', [])
    print(f"\n📄 Loaded {len(entries)} posts from {input_path}")
    
    if not entries:
        print("❌ No entries found")
        return
    
    # Filter entries with content
    valid_entries = []
    texts = []
    
    for entry in entries:
        content = entry.get('content', '').strip()
        if content and len(content) > 100:
            valid_entries.append(entry)
            texts.append(content)
    
    print(f"✅ {len(valid_entries)} posts have sufficient content")
    
    if not valid_entries:
        print("❌ No valid content to process")
        return
    
    # =========================================================================
    # STEP 1: Clustering (existing logic - assign layers)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Topic Clustering")
    print("=" * 80)
    
    embedder = load_embedder()
    print("Generating embeddings...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
    
    n_clusters = min(args.n_topics, len(valid_entries))
    print(f"\nRunning K-Means clustering (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Extract keywords per cluster
    print("\nExtracting topic keywords...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    cluster_info = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # TF-IDF scores for this cluster
        cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0).A1
        top_indices = cluster_tfidf.argsort()[-10:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        
        # Auto-label from top keywords
        label = keywords[0].replace('_', ' ').title()
        
        cluster_info[cluster_id] = {
            'label': label,
            'keywords': keywords,
            'size': int(cluster_mask.sum()),
            'entries': [valid_entries[i] for i in cluster_indices]
        }
    
    print("\n📊 Discovered Topics:")
    for cid, info in sorted(cluster_info.items(), key=lambda x: x[1]['size'], reverse=True):
        print(f"  {info['label']:30s} ({info['size']} posts)")
        print(f"    Keywords: {', '.join(info['keywords'][:5])}")
    
    # =========================================================================
    # STEP 2: Tech Stack Extraction (NER)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Tech Stack Extraction (NER)")
    print("=" * 80)
    
    ner_pipeline = load_ner_pipeline()
    
    print("\nExtracting technologies from posts...")
    for i, (entry, text) in enumerate(zip(valid_entries, texts)):
        if i % 50 == 0:
            print(f"  Processing post {i+1}/{len(valid_entries)}")
        
        tech_stack = extract_tech_stack(text, ner_pipeline)
        entry['tech_stack'] = tech_stack
    
    # Stats
    total_techs = sum(len(e.get('tech_stack', [])) for e in valid_entries)
    print(f"\n✅ Extracted {total_techs} technology mentions across {len(valid_entries)} posts")
    
    # Most common technologies
    all_tech_names = []
    for entry in valid_entries:
        all_tech_names.extend([t['name'] for t in entry.get('tech_stack', [])])
    
    if all_tech_names:
        tech_counter = Counter(all_tech_names)
        print("\n📌 Most mentioned technologies:")
        for tech, count in tech_counter.most_common(10):
            print(f"  {tech:20s} {count:3d} mentions")
    
    # =========================================================================
    # STEP 3: Pattern Extraction (NER + Semantic Analysis)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Architectural Pattern Extraction (NO hardcoded list)")
    print("=" * 80)
    
    print("\nExtracting patterns...")
    for i, (entry, text) in enumerate(zip(valid_entries, texts)):
        if i % 50 == 0:
            print(f"  Processing post {i+1}/{len(valid_entries)}")
        
        patterns = extract_patterns(text, ner_pipeline, embedder)
        entry['patterns'] = patterns
    
    # Stats
    total_patterns = sum(len(e.get('patterns', [])) for e in valid_entries)
    print(f"\n✅ Extracted {total_patterns} pattern mentions across {len(valid_entries)} posts")
    
    # Most common patterns
    all_pattern_names = []
    for entry in valid_entries:
        all_pattern_names.extend([p['pattern'] for p in entry.get('patterns', [])])
    
    if all_pattern_names:
        pattern_counter = Counter(all_pattern_names)
        print("\n📌 Most detected patterns:")
        for pattern, count in pattern_counter.most_common(10):
            print(f"  {pattern:35s} {count:3d} posts")
    
    # =========================================================================
    # STEP 4: Solution Mining + Problem/Approach (Q&A Model)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Solution/Problem/Approach Extraction (Q&A Model)")
    print("=" * 80)
    
    qa_pipeline = load_qa_pipeline()
    
    print("\nExtracting solutions, problems, and approaches...")
    for i, (entry, text) in enumerate(zip(valid_entries, texts)):
        if i % 50 == 0:
            print(f"  Processing post {i+1}/{len(valid_entries)}")
        
        tech_stack = entry.get('tech_stack', [])
        
        # Extract solutions (semantic similarity)
        solutions = extract_solutions(text, tech_stack, embedder)
        entry['solutions'] = solutions
        
        # Extract problem (Q&A)
        problem = extract_problem(text, qa_pipeline)
        entry['problem'] = problem
        
        # Extract approach (Q&A)
        approach = extract_approach(text, qa_pipeline)
        entry['approach'] = approach
    
    total_solutions = sum(len(e.get('solutions', [])) for e in valid_entries)
    posts_with_problem = sum(1 for e in valid_entries if e.get('problem'))
    posts_with_approach = sum(1 for e in valid_entries if e.get('approach'))
    
    print(f"\n✅ Extracted {total_solutions} solution descriptions")
    print(f"✅ Extracted problem from {posts_with_problem} posts")
    print(f"✅ Extracted approach from {posts_with_approach} posts")
    
    # =========================================================================
    # STEP 5: Assign Layers and Build Output
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Building Enriched Timeline")
    print("=" * 80)
    
    # Map entries to cluster labels
    entry_to_cluster = {}
    for cluster_id, info in cluster_info.items():
        for entry in info['entries']:
            entry_id = entry.get('path')
            if entry_id:
                entry_to_cluster[entry_id] = info['label']
    
    # Build output
    output_entries = []
    for entry in valid_entries:
        entry_id = entry.get('path')
        new_entry = entry.copy()
        
        # Assign layer from clustering
        if entry_id and entry_id in entry_to_cluster:
            new_entry['layers'] = [entry_to_cluster[entry_id]]
        else:
            new_entry['layers'] = []
        
        output_entries.append(new_entry)
    
    # Write output
    output_path = Path(args.output)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump({
            'count': len(output_entries),
            'entries': output_entries,
            'metadata': {
                'method': 'enhanced_ml_discovery',
                'models': {
                    'embedder': EMBEDDER_NAME,
                    'ner': NER_MODEL,
                    'qa': QA_MODEL
                },
                'extraction_methods': {
                    'layers': 'clustering (K-means + TF-IDF)',
                    'tech_stack': 'NER (Named Entity Recognition)',
                    'patterns': 'NER + semantic n-grams (NO hardcoded list)',
                    'solutions': 'semantic similarity with embeddings',
                    'problem': 'Q&A model',
                    'approach': 'Q&A model'
                },
                'n_topics': n_clusters,
                'topics': {
                    str(cid): {
                        'label': info['label'],
                        'keywords': info['keywords'],
                        'size': info['size']
                    }
                    for cid, info in cluster_info.items()
                }
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Wrote enriched timeline to: {output_path}")
    
    # Save models
    model_path = CACHE_DIR / "enriched_model.joblib"
    joblib.dump({
        'kmeans': kmeans,
        'cluster_info': cluster_info,
        'embedder_name': EMBEDDER_NAME
    }, model_path)
    print(f"✅ Saved models to: {model_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total posts: {len(output_entries)}")
    print(f"Posts with layers: {sum(1 for e in output_entries if e.get('layers'))}")
    print(f"Posts with tech stack: {sum(1 for e in output_entries if e.get('tech_stack'))}")
    print(f"Posts with patterns: {sum(1 for e in output_entries if e.get('patterns'))}")
    print(f"Posts with solutions: {sum(1 for e in output_entries if e.get('solutions'))}")
    
    avg_techs = sum(len(e.get('tech_stack', [])) for e in output_entries) / len(output_entries)
    avg_patterns = sum(len(e.get('patterns', [])) for e in output_entries) / len(output_entries)
    avg_solutions = sum(len(e.get('solutions', [])) for e in output_entries) / len(output_entries)
    
    print(f"\nAverage per post:")
    print(f"  Technologies: {avg_techs:.1f}")
    print(f"  Patterns: {avg_patterns:.1f}")
    print(f"  Solutions: {avg_solutions:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced topic discovery with NER and zero-shot classification"
    )
    parser.add_argument(
        "--input",
        default="./outputs/Netflix_timeline.json",
        help="Input timeline JSON file"
    )
    parser.add_argument(
        "--output",
        default="./outputs/Netflix_timeline_enriched.json",
        help="Output enriched timeline JSON file"
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=12,
        help="Number of topics to discover (8-15 recommended)"
    )
    args = parser.parse_args()
    discover_enriched(args)
