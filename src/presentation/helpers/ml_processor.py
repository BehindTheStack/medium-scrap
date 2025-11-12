"""
ML Processing Logic
Handles all machine learning extraction operations
"""

import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from rich.console import Console

from ..schemas.ml_schemas import MLDiscoveryData
from .progress_display import ProgressDisplay


class MLProcessor:
    """Handles ML extraction operations"""
    
    def __init__(self, console: Console):
        self.console = console
        self.embedder = None
        self.ner_pipeline = None
        self.qa_pipeline = None
        
    def load_models(self) -> float:
        """
        Load all ML models (embedder, NER, QA)
        Returns: loading time in seconds
        """
        # Add ML classifier path
        ml_path = Path(__file__).parent.parent.parent / 'ml_classifier'
        sys.path.insert(0, str(ml_path))
        
        from discover_enriched import (
            load_embedder, load_ner_pipeline, load_qa_pipeline
        )
        
        self.console.print("[dim]Loading models...[/dim]")
        start_time = time.time()
        
        self.embedder = load_embedder()
        self.ner_pipeline = load_ner_pipeline()
        self.qa_pipeline = load_qa_pipeline()
        
        load_time = time.time() - start_time
        self.console.print(f"[dim]✓ Models loaded in {load_time:.1f}s[/dim]")
        self.console.print()
        
        return load_time
    
    def cluster_topics(self, entries: List[Dict[str, Any]], texts: List[str]) -> Tuple[int, float]:
        """
        Cluster posts into topics using embeddings
        
        Args:
            entries: List of post entries to update with layers
            texts: List of post content texts
            
        Returns:
            Tuple of (number of clusters, processing time)
        """
        self.console.print("[cyan]📊 Step 1/6: Clustering for topics...[/cyan]")
        step_start = time.time()
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=False, batch_size=32)
        
        # Cluster
        n_clusters = min(8, len(entries))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Extract cluster keywords with TF-IDF
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        cluster_info = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0).A1
            top_indices = cluster_tfidf.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            label = keywords[0].replace('_', ' ').title()
            cluster_info[cluster_id] = {'label': label}
        
        # Assign layers to entries
        for i, entry in enumerate(entries):
            entry['layers'] = [cluster_info[cluster_labels[i]]['label']]
        
        step_time = time.time() - step_start
        self.console.print(f"[green]✓ Discovered {n_clusters} topics in {step_time:.1f}s[/green]")
        self.console.print()
        
        return n_clusters, step_time
    
    def extract_all_features(self, entries: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Extract all ML features (tech_stack, patterns, solutions, problem, approach)
        
        Args:
            entries: List of post entries to update
            
        Returns:
            Dictionary with extraction statistics
        """
        # Import extraction functions
        ml_path = Path(__file__).parent.parent.parent / 'ml_classifier'
        sys.path.insert(0, str(ml_path))
        
        from discover_enriched import (
            extract_tech_stack, extract_patterns, extract_solutions,
            extract_problem, extract_approach
        )
        
        total_posts = len(entries)
        
        self.console.print("[cyan]🤖 Processing ML extractions...[/cyan]")
        self.console.print()
        
        with ProgressDisplay.create_ml_progress(self.console) as progress:
            # Create tasks
            task_tech = progress.add_task("[cyan]2/6 Tech Stack      ", total=total_posts)
            task_patterns = progress.add_task("[cyan]3/6 Patterns        ", total=total_posts)
            task_solutions = progress.add_task("[cyan]4/6 Solutions       ", total=total_posts)
            task_problems = progress.add_task("[cyan]5/6 Problems        ", total=total_posts)
            task_approaches = progress.add_task("[cyan]6/6 Approaches      ", total=total_posts)
            
            # Extract Tech Stack
            for entry in entries:
                tech_stack = extract_tech_stack(entry['content'], self.ner_pipeline)
                entry['tech_stack'] = tech_stack
                progress.update(task_tech, advance=1)
            
            # Extract Patterns
            for entry in entries:
                patterns = extract_patterns(entry['content'], self.ner_pipeline, self.embedder)
                entry['patterns'] = patterns
                progress.update(task_patterns, advance=1)
            
            # Extract Solutions
            for entry in entries:
                tech_stack = entry.get('tech_stack', [])
                solutions = extract_solutions(entry['content'], tech_stack, self.embedder)
                entry['solutions'] = solutions
                progress.update(task_solutions, advance=1)
            
            # Extract Problems
            for entry in entries:
                problem = extract_problem(entry['content'], self.qa_pipeline)
                entry['problem'] = problem
                progress.update(task_problems, advance=1)
            
            # Extract Approaches
            for entry in entries:
                approach = extract_approach(entry['content'], self.qa_pipeline)
                entry['approach'] = approach
                progress.update(task_approaches, advance=1)
        
        # Calculate statistics
        stats = {
            'total_techs': sum(len(e.get('tech_stack', [])) for e in entries),
            'total_patterns': sum(len(e.get('patterns', [])) for e in entries),
            'total_solutions': sum(len(e.get('solutions', [])) for e in entries),
            'posts_with_problem': sum(1 for e in entries if e.get('problem')),
            'posts_with_approach': sum(1 for e in entries if e.get('approach')),
        }
        
        self.console.print()
        self.console.print(f"[green]✓ Extracted:[/green]")
        self.console.print(f"  • {stats['total_techs']} tech stack items")
        self.console.print(f"  • {stats['total_patterns']} patterns")
        self.console.print(f"  • {stats['total_solutions']} solutions")
        self.console.print(f"  • {stats['posts_with_problem']} problems")
        self.console.print(f"  • {stats['posts_with_approach']} approaches")
        self.console.print()
        
        return stats
    
    def process_posts(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Complete ML processing pipeline for posts
        
        Args:
            entries: List of post entries with 'content' field
            
        Returns:
            Dictionary with processing statistics
        """
        if not entries:
            return {}
        
        # Extract texts
        texts = [e['content'] for e in entries]
        
        # Step 1: Clustering
        n_clusters, cluster_time = self.cluster_topics(entries, texts)
        
        # Steps 2-6: Feature extraction
        stats = self.extract_all_features(entries)
        stats['n_clusters'] = n_clusters
        stats['cluster_time'] = cluster_time
        
        return stats
