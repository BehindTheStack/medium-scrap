"""
Manual topic labeling based on cluster analysis.
After reviewing the discovered clusters, assign meaningful labels.
"""
from pathlib import Path
import json

# Based on keyword analysis and sample titles, here are better labels:
MANUAL_TOPIC_LABELS = {
    "Api/Rx": "backend-apis",              # Reactive programming, API design
    "Video/Quality": "video-streaming",     # Encoding, quality, playback
    "Security/Open": "platform-tooling",    # Open source tools, infrastructure
    "Data/Cloud": "data-infrastructure",    # Largest cluster - data systems, infra
    "Data/Time": "observability",           # Monitoring, metrics, auto-scaling
    "Data/Engineering": "engineering-culture",  # General engineering, practices
    "Data/Performance": "performance",      # Performance optimization, caching
    "Data/Service": "distributed-systems",  # Cassandra, databases, microservices
    "Aws/Cloud": "cloud-infrastructure",    # AWS-specific, cloud migration
}

def relabel_timeline():
    input_path = Path("outputs/Netflix_timeline_discovered.json")
    output_path = Path("outputs/Netflix_timeline_refined.json")
    
    with input_path.open('r') as f:
        data = json.load(f)
    
    # Relabel
    for entry in data['entries']:
        if entry.get('layers') and len(entry['layers']) > 0:
            old_label = entry['layers'][0]
            new_label = MANUAL_TOPIC_LABELS.get(old_label, old_label)
            entry['layers'] = [new_label]
    
    # Write
    with output_path.open('w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Relabeled timeline written to {output_path}")
    
    # Show distribution
    from collections import Counter
    label_counts = Counter()
    for e in data['entries']:
        if e.get('layers'):
            label_counts[e['layers'][0]] += 1
    
    print("\nNew label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:30s} {count:3d} posts")

if __name__ == "__main__":
    relabel_timeline()
