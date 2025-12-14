
"""
ML Processing Logic
Handles all machine learning extraction operations

Supports two extraction modes:
1. Legacy: BERT NER + n-grams + QA (original approach)
2. Modern: GLiNER + Semantic Patterns + Optional LLM (2024 approach)

------------------------------------------------------------
PATTERNS vs TOPIC LAYERS
------------------------------------------------------------
• Architecture Patterns: 28 específicos, categorizados e centralizados em PATTERN_TO_LAYER (importado de tech_extractor.py). Usados para inferir camada arquitetural a partir de padrões reconhecidos no texto.
• Topic Layers: 8 genéricos, baseados em clustering de tópicos (Data & ML, Backend & APIs, Infrastructure, Frontend & UI, Database & Storage, Streaming & Events, Observability, Security). Usados para agrupamento automático via KMeans/TF-IDF.

→ Patterns = taxonomia explícita (regra de negócio)
→ Topic Layers = agrupamento automático (clustering)
------------------------------------------------------------
"""

import re
import time
from typing import List, Dict, Any, Tuple, Optional, Callable
from pathlib import Path
import sys

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from rich.console import Console

from .progress_display import ProgressDisplay
from .text_cleaner import clean_markdown
from ...ml_classifier.tech_extractor import PATTERN_TO_LAYER


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

        # Extract and clean texts from entries once and cache on the entry dict
        for entry in entries:
            if '_clean' not in entry:
                entry['_clean'] = clean_markdown(entry.get('content_markdown', ''))
        texts = [entry.get('_clean', '') for entry in entries]
        
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
                content_clean = entry.get('_clean') or clean_markdown(entry.get('content_markdown', ''))
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
                content_clean = entry.get('_clean') or clean_markdown(entry.get('content_markdown', ''))
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
                content_clean = entry.get('_clean') or clean_markdown(entry.get('content_markdown', ''))
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
                content_clean = entry.get('_clean') or clean_markdown(entry.get('content_markdown', ''))
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
                content_clean = entry.get('_clean') or clean_markdown(entry.get('content_markdown', ''))
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
    
    def process_posts_optimized(
        self,
        all_entries: List[Dict[str, Any]],
        save_callback: Optional[Callable] = None,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Process posts with HYBRID approach:
        - Clustering GLOBAL (TF-IDF needs all docs)
        - Extractions in BATCHES (memory efficient)
        
        Args:
            all_entries: ALL posts to process
            save_callback: Function to save individual post ML data
            batch_size: Posts per batch for extraction (default: 50)
            
        Returns:
            Dictionary with processing statistics
        """
        if not all_entries:
            self.console.print("[yellow]No entries to process[/yellow]")
            return {}
        
        total_start = time.time()
        
        # PHASE 1: GLOBAL CLUSTERING (needs all posts together)
        self.console.print(f"\n[bold cyan]Phase 1: Global Clustering ({len(all_entries)} posts)[/bold cyan]")
        texts = [e.get('content', '') for e in all_entries]
        self.cluster_topics(all_entries, texts)
        
        # PHASE 2: BATCH PROCESSING (efficient extraction)
        self.console.print(f"\n[bold cyan]Phase 2: Batch ML Extraction (batches of {batch_size})[/bold cyan]")
        
        total_stats = {
            'tech_stack': 0,
            'patterns': 0,
            'solutions': 0,
            'problems': 0,
            'approaches': 0
        }
        
        num_batches = (len(all_entries) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_entries))
            batch = all_entries[start_idx:end_idx]
            
            self.console.print(
                f"\n[cyan]Batch {batch_idx + 1}/{num_batches}: "
                f"Posts {start_idx + 1}-{end_idx}[/cyan]"
            )
            
            # Extract features for this batch
            batch_stats = self._process_batch_extractions(batch)
            
            # Accumulate stats
            for key in total_stats:
                total_stats[key] += batch_stats.get(key, 0)
            
            # Save batch to database
            if save_callback:
                self._save_batch_results(batch, save_callback)
            
            # Clear GPU cache between batches
            self._clear_gpu_cache()
        
        total_time = time.time() - total_start
        
        # Print final summary
        self.console.print(f"\n[bold green]✅ Completed in {total_time:.1f}s[/bold green]")
        self.console.print(f"[dim]Average: {total_time/len(all_entries):.2f}s per post[/dim]\n")
        
        return total_stats
    
    def _process_batch_extractions(self, batch: List[Dict[str, Any]]) -> Dict[str, int]:
        """Extract all features for a batch of posts"""
        from discover_enriched import (
            extract_tech_stack,
            extract_patterns,
            extract_solutions,
            extract_problem,
            extract_approach
        )
        
        stats = {
            'tech_stack': 0,
            'patterns': 0,
            'solutions': 0,
            'problems': 0,
            'approaches': 0
        }
        
        # Step 2: Tech Stack (NER - batch friendly)
        self.console.print("[cyan]  → Tech Stack (NER)...[/cyan]")
        for entry in batch:
            tech = extract_tech_stack(entry.get('content', ''), self.ner_pipeline)
            entry['tech_stack'] = tech
            if tech:
                stats['tech_stack'] += 1
        
        # Step 3: Patterns (NER - batch friendly)
        self.console.print("[cyan]  → Patterns...[/cyan]")
        for entry in batch:
            patterns = extract_patterns(entry.get('content', ''), self.ner_pipeline, self.embedder)
            entry['patterns'] = patterns
            if patterns:
                stats['patterns'] += 1
        
        # Step 4: Solutions (embeddings - controlled batching)
        self.console.print("[cyan]  → Solutions...[/cyan]")
        for entry in batch:
            tech_stack = entry.get('tech_stack', [])
            solutions = extract_solutions(
                entry.get('content', ''),
                tech_stack,
                self.embedder
            )
            entry['solutions'] = solutions
            if solutions:
                stats['solutions'] += 1
        
        # Step 5: Problems (Q&A - sequential but fast)
        self.console.print("[cyan]  → Problems...[/cyan]")
        for entry in batch:
            problem = extract_problem(entry.get('content', ''), self.qa_pipeline)
            entry['problem'] = problem
            if problem:
                stats['problems'] += 1
        
        # Step 6: Approaches (Q&A - sequential but fast)
        self.console.print("[cyan]  → Approaches...[/cyan]")
        for entry in batch:
            approach = extract_approach(entry.get('content', ''), self.qa_pipeline)
            entry['approach'] = approach
            if approach:
                stats['approaches'] += 1
        
        return stats
    
    def _save_batch_results(
        self,
        batch: List[Dict[str, Any]],
        save_callback: Callable
    ) -> None:
        """Save batch results to database"""
        self.console.print("[cyan]  → Saving batch...[/cyan]")
        
        for entry in batch:
            ml_data = {
                'layers': entry.get('layers', []),
                'tech_stack': entry.get('tech_stack', []),
                'patterns': entry.get('patterns', []),
                'solutions': entry.get('solutions', []),
                'problem': entry.get('problem'),
                'approach': entry.get('approach'),
            }
            save_callback(entry['id'], ml_data)
        
        self.console.print(f"[green]  ✓ Saved {len(batch)} posts[/green]")


class ModernMLProcessor:
    """
    Modern ML Processor using hybrid extraction pipeline (2024 approach)
    
    Uses:
    - GLiNER: Zero-shot NER for custom technical entities
    - Semantic Pattern Classification: Embedding-based architecture detection
    - Optional LLM: Deep extraction via Ollama (Qwen2.5-7B)
    
    Much better than legacy BERT NER for technical content.
    """
    
    def __init__(self, console: Console):
        self.console = console
        self.pipeline = None
        self._loaded = False
        
    def load_models(
        self,
        use_gliner: bool = True,
        use_patterns: bool = True,
        use_llm: bool = False,
        llm_model: str = "qwen2.5:14b"
    ) -> float:
        """
        Load the modern extraction pipeline.
        
        Args:
            use_gliner: Enable GLiNER tech extraction
            use_patterns: Enable pattern classification
            use_llm: Enable LLM deep extraction (slower)
            llm_model: Ollama model name
            
        Returns:
            Loading time in seconds
        """
        if self._loaded:
            return 0.0
            
        start_time = time.time()
        
        # Import the modern pipeline
        ml_path = Path(__file__).parent.parent.parent / 'ml_classifier'
        sys.path.insert(0, str(ml_path))
        
        from tech_extractor import TechExtractionPipeline
        
        self.console.print("[dim]Loading modern ML pipeline...[/dim]")
        
        self.pipeline = TechExtractionPipeline(
            use_gliner=use_gliner,
            use_patterns=use_patterns,
            use_llm=use_llm,
            llm_model=llm_model
        )
        
        self._loaded = True
        load_time = time.time() - start_time
        
        self.console.print(f"[dim]✓ Modern pipeline ready in {load_time:.1f}s[/dim]")
        return load_time
    
    def process_posts(
        self,
        entries: List[Dict[str, Any]],
        save_callback: Optional[Callable[[str, Dict], None]] = None,
        use_llm: bool = False
    ) -> Dict[str, Any]:
        """
        Process posts with modern hybrid extraction.
        
        Args:
            entries: List of post entries
            save_callback: Optional callback(post_id, ml_data) to save
            use_llm: Enable LLM for deep extraction
            
        Returns:
            Processing statistics
        """
        if not entries:
            return {'total': 0}
        
        # Load models if needed
        if not self._loaded:
            self.load_models(use_llm=use_llm)
        
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
        from tech_extractor import extraction_to_dict
        
        total_posts = len(entries)
        
        stats = {
            'total': total_posts,
            'tech_extracted': 0,
            'patterns_detected': 0,
            'problems_extracted': 0,
            'total_tech_items': 0,
            'total_patterns': 0
        }
        
        self.console.print(f"\n[bold cyan]🔬 Modern ML Extraction ({total_posts} posts)[/bold cyan]")
        self.console.print("[dim]Using: GLiNER + Semantic Patterns" + 
                          (" + LLM" if use_llm else "") + "[/dim]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Processing", total=total_posts)
            
            for entry in entries:
                post_id = entry.get('id', '')
                content = entry.get('content_markdown', '') or entry.get('content', '')
                
                if not content or len(content) < 100:
                    progress.update(task, advance=1)
                    continue
                
                # Extract using modern pipeline
                result = self.pipeline.process(post_id, content)
                
                # Convert to legacy format for compatibility
                entry['tech_stack'] = [
                    {'name': t.name, 'type': t.category, 'score': t.confidence}
                    for t in result.tech_stack
                ]
                entry['patterns'] = [
                    {'pattern': p.name, 'confidence': p.confidence}
                    for p in result.patterns
                ]
                entry['solutions'] = result.solutions
                entry['problem'] = result.problems[0] if result.problems else None
                entry['approach'] = result.key_decisions[0] if result.key_decisions else None
                entry['layers'] = self._infer_layer_from_patterns(result.patterns)
                
                # Update stats
                if result.tech_stack:
                    stats['tech_extracted'] += 1
                    stats['total_tech_items'] += len(result.tech_stack)
                if result.patterns:
                    stats['patterns_detected'] += 1
                    stats['total_patterns'] += len(result.patterns)
                if result.problems:
                    stats['problems_extracted'] += 1
                
                # Save if callback provided
                if save_callback:
                    ml_data = {
                        'layers': entry['layers'],
                        'tech_stack': entry['tech_stack'],
                        'patterns': entry['patterns'],
                        'solutions': entry['solutions'],
                        'problem': entry['problem'],
                        'approach': entry['approach'],
                    }
                    save_callback(post_id, ml_data)
                
                progress.update(task, advance=1)
        
        # Print summary
        self.console.print(f"\n[green]✅ Extraction Complete[/green]")
        self.console.print(f"  📦 Tech Stack: {stats['total_tech_items']} items from {stats['tech_extracted']} posts")
        self.console.print(f"  🏗️  Patterns: {stats['total_patterns']} detected in {stats['patterns_detected']} posts")
        self.console.print(f"  ❓ Problems: {stats['problems_extracted']} posts")
        
        return stats
    
    def _infer_layer_from_patterns(self, patterns) -> List[str]:
        """Infer architecture layer from detected patterns"""
        if not patterns:
            return ['General']
        
        # Map patterns to layers (aligned with tech_extractor.py 28 patterns)
        # Categories: Distributed Systems, Backend & APIs, Cloud Infrastructure,
        #             Performance & Resilience, Observability, Data Infrastructure, ML/AI Platform
        layers = []
        for p in patterns:
            pattern_name = p.name if hasattr(p, 'name') else p.get('pattern', '')
            layer = PATTERN_TO_LAYER.get(pattern_name)
            if layer and layer not in layers:
                layers.append(layer)
        
        return layers if layers else ['General']
    
    def _clear_gpu_cache(self):
        """Clear GPU cache"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
