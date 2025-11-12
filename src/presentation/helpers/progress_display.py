"""
Progress Display Helpers
"""

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn


class ProgressDisplay:
    """Helper for displaying progress bars"""
    
    @staticmethod
    def create_ml_progress(console: Console) -> Progress:
        """Create progress bar for ML extraction"""
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False
        )
    
    @staticmethod
    def create_simple_progress(console: Console) -> Progress:
        """Create simple progress bar for saving/loading"""
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False
        )
