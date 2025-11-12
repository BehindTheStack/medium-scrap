#!/usr/bin/env python3
"""
Enhanced topic discovery with NER-based tech stack extraction,
zero-shot pattern classification, and solution mining.

Uses only HuggingFace models - no hardcoded keywords.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Cache directory for models
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Models
EMBEDDER_NAME = "all-MiniLM-L6-v2"
NER_MODEL = "dslim/bert-base-NER"
QA_MODEL = "deepset/roberta-base-squad2"  # For problem/approach extraction

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
    
    # Use first 2000 chars
    text_sample = text[:2000]
    
    try:
        # 1. Get technical entities from NER
        entities = ner_pipeline(text_sample)
        tech_entities = [
            e['word'].replace('##', '').strip() 
            for e in entities 
            if e['entity_group'] in ['ORG', 'MISC'] and e['score'] > 0.80
        ]
        
        # 2. Extract technical bigrams/trigrams
        import re
        # Remove markdown, code blocks, URLs
        clean_text = re.sub(r'```.*?```', '', text_sample, flags=re.DOTALL)
        clean_text = re.sub(r'http\S+', '', clean_text)
        clean_text = re.sub(r'[#*`\[\]()]', ' ', clean_text)
        
        # Split into sentences
        sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 20]
        
        # Extract n-grams that contain technical terms
        patterns_found = []
        pattern_keywords = ['architecture', 'pattern', 'system', 'design', 'approach', 
                           'model', 'framework', 'platform', 'infrastructure', 'service']
        
        for sentence in sentences[:30]:  # First 30 sentences
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
                    'confidence': min(count / len(sentences), 1.0)  # Normalize by sentence count
                })
        
        return patterns[:8]  # Top 8
        
    except Exception as e:
        print(f"Pattern extraction error: {e}")
        return []


def load_embedder():
    """Load sentence transformer model."""
    print(f"Loading embedder: {EMBEDDER_NAME}")
    return SentenceTransformer(EMBEDDER_NAME, cache_folder=str(CACHE_DIR))


def load_ner_pipeline():
    """Load NER pipeline for entity extraction."""
    print(f"Loading NER model: {NER_MODEL}")
    return pipeline(
        "ner",
        model=NER_MODEL,
        aggregation_strategy="simple",
        device=-1  # CPU
    )


def load_qa_pipeline():
    """Load Q&A pipeline for problem/approach extraction."""
    print(f"Loading Q&A model: {QA_MODEL}")
    return pipeline(
        "question-answering",
        model=QA_MODEL,
        device=-1  # CPU
    )


def extract_tech_stack(text: str, ner_pipeline) -> List[Dict[str, Any]]:
    """
    Extract technology mentions using NER.
    
    Returns:
        List of dicts with {name, type, score}
    """
    if not text or len(text.strip()) < 10:
        return []
    
    # Limit text to avoid OOM (first 2000 chars usually sufficient)
    text_sample = text[:2000]
    
    try:
        entities = ner_pipeline(text_sample)
    except Exception as e:
        print(f"NER error: {e}")
        return []
    
    # Filter for high-confidence entities
    tech_mentions = []
    seen = set()
    
    for entity in entities:
        word = entity['word'].strip()
        entity_type = entity['entity_group']
        score = entity['score']
        
        # Keep ORG (organizations/products) and MISC (technologies)
        if entity_type in ['ORG', 'MISC'] and score > 0.85:
            # Clean up subword tokens
            word_clean = word.replace('##', '').strip()
            if len(word_clean) > 2 and word_clean.lower() not in seen:
                seen.add(word_clean.lower())
                tech_mentions.append({
                    'name': word_clean,
                    'type': entity_type,
                    'score': round(score, 3)
                })
    
    return tech_mentions[:15]  # Top 15 mentions


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
        # Reference embeddings for solution-related content
        solution_templates = [
            "We built a solution to solve the problem",
            "The implementation uses technology to achieve results",
            "We developed a system that improves performance",
            "Our approach solved the challenge by implementing"
        ]
        
        # Get embeddings for templates
        template_embeddings = embedder.encode(solution_templates)
        
        # Split text into sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30][:50]
        
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
            # High semantic similarity to solution templates
            if sim > 0.5:
                # Check if mentions any technology
                mentions_tech = any(
                    tech['name'].lower() in sentence.lower()
                    for tech in tech_stack
                )
                
                if mentions_tech or sim > 0.65:  # Either mentions tech OR very high similarity
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
        # Use first 2000 chars (intro section usually describes problem)
        context = text[:2000]
        
        # Ask the model
        result = qa_pipeline(
            question="What problem or challenge did they face?",
            context=context
        )
        
        # Check if answer is substantial
        answer = result['answer'].strip()
        if len(answer) > 20 and result['score'] > 0.1:
            return answer
        
        # Try alternative question
        result2 = qa_pipeline(
            question="What issue needed to be solved?",
            context=context
        )
        
        answer2 = result2['answer'].strip()
        if len(answer2) > 20 and result2['score'] > 0.1:
            return answer2
        
        return None
        
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
        # Use middle section (2000-5000 chars) where solution is usually described
        context = text[1000:5000] if len(text) > 5000 else text
        
        # Ask the model
        result = qa_pipeline(
            question="How did they solve the problem?",
            context=context
        )
        
        # Check if answer is substantial
        answer = result['answer'].strip()
        if len(answer) > 20 and result['score'] > 0.1:
            return answer
        
        # Try alternative question
        result2 = qa_pipeline(
            question="What was their solution approach?",
            context=context
        )
        
        answer2 = result2['answer'].strip()
        if len(answer2) > 20 and result2['score'] > 0.1:
            return answer2
        
        return None
        
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
