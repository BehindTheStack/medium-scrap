"""
ML Processing Logic
Handles all machine learning extraction operations
"""

import re
import time
from typing import List, Dict, Any, Tuple, Optional, Callable
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from rich.console import Console

from ..schemas.ml_schemas import MLDiscoveryData
from .progress_display import ProgressDisplay
from .text_cleaner import clean_markdown, chunk_text


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
        # Suppress transformer warnings about unused weights
        import warnings
        import transformers
        transformers.logging.set_verbosity_error()
        warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
        
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
    
    # Note: text cleaning moved to src/presentation/helpers/text_cleaner.py
    # We import and use clean_markdown() there so the cleaning logic is centralized
    
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
        
        # Clean texts for clustering (preserve structure for better TF-IDF)
        cleaned_texts = [clean_markdown(t, preserve_structure=True) for t in texts]
        
        # Filter out empty or very short texts
        valid_indices = []
        valid_texts = []
        for i, txt in enumerate(cleaned_texts):
            if len(txt.strip()) > 50:  # Minimum meaningful length
                valid_indices.append(i)
                valid_texts.append(txt)
        
        if len(valid_texts) < 2:
            # Not enough text to cluster
            self.console.print(f"[yellow]⚠ Only {len(valid_texts)} posts have sufficient text, assigning default layer[/yellow]")
            for entry in entries:
                entry['layers'] = ['General']
            return {
                'num_clusters': 1,
                'cluster_time': time.time() - step_start
            }
        
        # Generate embeddings
        embeddings = self.embedder.encode(valid_texts, show_progress_bar=False, batch_size=32)
        
        # Cluster (adjust n_clusters based on data size)
        n_clusters = min(8, max(2, len(valid_texts) // 5))  # At least 5 posts per cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Extract cluster keywords with TF-IDF
        try:
            vectorizer = TfidfVectorizer(
                max_features=500, 
                stop_words='english', 
                ngram_range=(1, 3),  # Include trigrams for better context
                min_df=1,
                max_df=0.95,
                token_pattern=r'\b[a-zA-Z]{3,}\b'  # Words with 3+ letters only
            )
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Architecture layer mapping based on keywords
            arch_patterns = {
                'Data & ML': ['data', 'machine learning', 'ml', 'model', 'training', 'dataset', 'pipeline', 'spark', 'hadoop', 'airflow'],
                'Backend & APIs': ['api', 'backend', 'service', 'microservice', 'rest', 'grpc', 'graphql', 'server', 'endpoint'],
                'Infrastructure': ['infrastructure', 'deployment', 'kubernetes', 'docker', 'cloud', 'aws', 'gcp', 'terraform', 'devops'],
                'Frontend & UI': ['frontend', 'ui', 'ux', 'react', 'vue', 'angular', 'web', 'mobile', 'app', 'interface'],
                'Database & Storage': ['database', 'storage', 'sql', 'nosql', 'cache', 'redis', 'postgres', 'mongo', 'cassandra'],
                'Streaming & Events': ['streaming', 'event', 'kafka', 'real time', 'realtime', 'live', 'message', 'queue'],
                'Observability': ['monitoring', 'logging', 'metrics', 'telemetry', 'observability', 'tracing', 'alerting', 'grafana', 'prometheus'],
                'Security': ['security', 'authentication', 'authorization', 'encryption', 'auth', 'oauth', 'identity'],
            }
            
            cluster_info = {}
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0).A1
                top_indices = cluster_tfidf.argsort()[-20:][::-1]  # Get more keywords for better matching
                keywords = [feature_names[i] for i in top_indices if i < len(feature_names)]
                
                # Try to match to architecture layer
                label = None
                best_match_score = 0
                keywords_lower = [k.lower() for k in keywords]
                
                for arch_layer, patterns in arch_patterns.items():
                    # Count matches
                    matches = sum(1 for pattern in patterns if any(pattern in kw for kw in keywords_lower))
                    if matches > best_match_score:
                        best_match_score = matches
                        label = arch_layer
                
                # Fallback to TF-IDF keywords if no good match
                if not label or best_match_score < 2:
                    # Use top 2-3 meaningful keywords
                    meaningful_kw = [kw for kw in keywords[:10] 
                                    if len(kw) > 4 and kw.lower() not in {'using', 'build', 'system', 'service', 'application'}]
                    if meaningful_kw:
                        label = ' & '.join(meaningful_kw[:2]).title()
                    else:
                        label = f"Topic {cluster_id + 1}"
                    
                cluster_info[cluster_id] = {'label': label, 'keywords': keywords}
                
        except ValueError as e:
            # TF-IDF failed - use generic labels with word frequency fallback
            self.console.print(f"[yellow]⚠ TF-IDF failed, using word frequency for labels[/yellow]")
            cluster_info = {}
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_texts = [valid_texts[i] for i in range(len(cluster_labels)) if cluster_mask[i]]
                
                # Simple word frequency
                from collections import Counter
                words = []
                for txt in cluster_texts:
                    # Extract words (3+ letters, alphanumeric)
                    words.extend(re.findall(r'\b[a-zA-Z]{3,}\b', txt.lower()))
                
                # Remove common stop words manually
                stop_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 
                             'was', 'were', 'have', 'has', 'had', 'been', 'our', 'their'}
                words = [w for w in words if w not in stop_words]
                
                if words:
                    common = Counter(words).most_common(5)
                    keywords = [w for w, _ in common]
                    label = ' & '.join(keywords[:2]).title()
                else:
                    keywords = []
                    label = f"Topic {cluster_id + 1}"
                
                cluster_info[cluster_id] = {'label': label, 'keywords': keywords}
        
        # Assign layers to all entries (including those skipped)
        cluster_idx = 0
        for i, entry in enumerate(entries):
            if i in valid_indices:
                entry['layers'] = [cluster_info[cluster_labels[cluster_idx]]['label']]
                cluster_idx += 1
            else:
                # Assign to a default or most common cluster
                entry['layers'] = [cluster_info[0]['label']] if cluster_info else ['General']
        
        step_time = time.time() - step_start
        self.console.print(f"[green]✓ Discovered {n_clusters} topics in {step_time:.1f}s[/green]")
        self.console.print()
        
        return {
            'num_clusters': n_clusters,
            'cluster_time': step_time
        }
    
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
    
    def process_posts(self, entries: List[Dict[str, Any]], 
                     save_callback: Optional[Callable[[Dict], None]] = None) -> Dict[str, int]:
        """
        Complete ML pipeline: clustering + feature extraction
        
        Args:
            entries: List of post entries to process
            save_callback: Optional callback to save each post after processing
                         Called with (post_id, ml_data) after each post is classified
        
        Returns:
            Processing statistics
        """
        if not entries:
            return {'total': 0}
        
        self.load_models()
        
        # Extract and clean texts from entries (centralized helper)
        texts = [clean_markdown(entry.get('content_markdown', '')) for entry in entries]
        
        # Step 1: Clustering for layers
        cluster_stats = self.cluster_topics(entries, texts)
        
        # Step 2: Extract all features (with incremental save if callback provided)
        if save_callback:
            # Process and save one by one
            extraction_stats = self._extract_with_callback(entries, save_callback)
        else:
            # Original behavior: extract all, then return
            extraction_stats = self.extract_all_features(entries)
        
        return {
            **cluster_stats,
            **extraction_stats,
            'total': len(entries)
        }
    
    def _extract_with_callback(self, entries: List[Dict[str, Any]], 
                               save_callback: Callable[[str, Dict], None]) -> Dict[str, int]:
        """Extract features and save incrementally with progress bars"""
        from discover_enriched import (
            extract_tech_stack, extract_patterns, extract_solutions,
            extract_problem, extract_approach
        )
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
        
        total_posts = len(entries)
        
        # Filter valid entries first
        valid_entries = [e for e in entries if e.get('content_markdown', '') and len(e.get('content_markdown', '')) >= 100]
        
        if not valid_entries:
            self.console.print("[yellow]⚠ No valid entries with sufficient content[/yellow]")
            return {
                'tech_stack_extracted': 0,
                'patterns_extracted': 0,
                'solutions_extracted': 0,
                'problems_extracted': 0,
                'approaches_extracted': 0,
            }
        
        self.console.print("[cyan]🤖 Processing ML extractions...[/cyan]")
        self.console.print()
        
        stats = {
            'tech_stack_extracted': 0,
            'patterns_extracted': 0,
            'solutions_extracted': 0,
            'problems_extracted': 0,
            'approaches_extracted': 0,
        }
        
        # Step 2: Tech Stack (NER) - with progress bar
        self.console.print(f"[cyan]📊 Step 2/6: Tech Stack (NER)...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Extracting tech stack", total=len(valid_entries))
            for entry in valid_entries:
                content_clean = clean_markdown(entry['content_markdown'])
                tech_stack = extract_tech_stack(content_clean, self.ner_pipeline)
                entry['tech_stack'] = tech_stack
                if tech_stack:
                    stats['tech_stack_extracted'] += 1
                progress.update(task, advance=1)
        self.console.print(f"[green]✓ Extracted tech stack from {stats['tech_stack_extracted']} posts[/green]")
        self.console.print()
        self._clear_gpu_cache()
        
        # Step 3: Patterns (NER + ngrams) - with progress bar
        self.console.print(f"[cyan]📊 Step 3/6: Patterns (NER + Semantic)...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Extracting patterns", total=len(valid_entries))
            for entry in valid_entries:
                content_clean = clean_markdown(entry['content_markdown'])
                patterns = extract_patterns(content_clean, self.ner_pipeline, self.embedder)
                entry['patterns'] = patterns
                if patterns:
                    stats['patterns_extracted'] += 1
                progress.update(task, advance=1)
        self.console.print(f"[green]✓ Extracted patterns from {stats['patterns_extracted']} posts[/green]")
        self.console.print()
        self._clear_gpu_cache()
        
        # Step 4: Solutions (Embeddings) - with progress bar
        self.console.print(f"[cyan]📊 Step 4/6: Solutions (Semantic Similarity)...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Extracting solutions", total=len(valid_entries))
            for entry in valid_entries:
                content_clean = clean_markdown(entry['content_markdown'])
                tech_stack = entry.get('tech_stack', [])
                solutions = extract_solutions(content_clean, tech_stack, self.embedder)
                entry['solutions'] = solutions
                if solutions:
                    stats['solutions_extracted'] += 1
                progress.update(task, advance=1)
        self.console.print(f"[green]✓ Extracted solutions from {stats['solutions_extracted']} posts[/green]")
        self.console.print()
        self._clear_gpu_cache()
        
        # Step 5: Problems (Q&A) - with progress bar
        self.console.print(f"[cyan]📊 Step 5/6: Problems (Q&A Model)...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Extracting problems", total=len(valid_entries))
            for entry in valid_entries:
                content_clean = clean_markdown(entry['content_markdown'])
                problem = extract_problem(content_clean, self.qa_pipeline)
                entry['problem'] = problem
                if problem:
                    stats['problems_extracted'] += 1
                progress.update(task, advance=1)
        self.console.print(f"[green]✓ Extracted problems from {stats['problems_extracted']} posts[/green]")
        self.console.print()
        self._clear_gpu_cache()
        
        # Step 6: Approaches (Q&A) - with progress bar
        self.console.print(f"[cyan]📊 Step 6/6: Approaches (Q&A Model)...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Extracting approaches", total=len(valid_entries))
            for entry in valid_entries:
                content_clean = clean_markdown(entry['content_markdown'])
                approach = extract_approach(content_clean, self.qa_pipeline)
                entry['approach'] = approach
                if approach:
                    stats['approaches_extracted'] += 1
                progress.update(task, advance=1)
        self.console.print(f"[green]✓ Extracted approaches from {stats['approaches_extracted']} posts[/green]")
        self.console.print()
        self._clear_gpu_cache()
        
        # Save all processed entries
        self.console.print(f"[cyan]💾 Saving {len(valid_entries)} processed posts...[/cyan]")
        saved = 0
        for entry in valid_entries:
            ml_data = {
                'layers': entry.get('layers', []),
                'tech_stack': entry.get('tech_stack', []),
                'patterns': entry.get('patterns', []),
                'solutions': entry.get('solutions', []),
                'problem': entry.get('problem'),
                'approach': entry.get('approach'),
            }
            save_callback(entry['id'], ml_data)
            saved += 1
            if saved % 10 == 0:
                self.console.print(f"[dim]  Saved {saved}/{len(valid_entries)}...[/dim]", end='\r')
        
        self.console.print(f"[green]✓ Saved {saved} posts to database[/green]")
        self.console.print()
        
        return stats
    
    def _clear_gpu_cache(self):
        """Clear GPU cache to avoid memory issues"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
