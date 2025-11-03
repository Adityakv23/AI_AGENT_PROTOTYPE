"""
Command-line interface for the AI Agent Prototype
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from pathlib import Path
import json

from config import USER_INFO, OUTPUT_DIR
from agents import AIAgent
from evaluation import Evaluator
from fine_tuning import FineTuner
from data_preparation import DataPreparator
from rag_system import RAGSystem

app = typer.Typer(help="AI Agent Prototype - Academic Paper Summarization System")
console = Console()


@app.command()
def welcome():
    """Display welcome message and project information"""
    welcome_text = f"""
    Welcome to AI Agent Prototype!
    
    Project: Academic Paper Summarization Agent
    {USER_INFO['name']}
    {USER_INFO['university']}
    {USER_INFO['department']}
    
    This agent automates the task of summarizing academic papers using:
    • Fine-tuned LoRA model for task specialization
    • Multi-agent collaboration (Planner + Executor)
    • RAG system for enhanced context understanding
    • Comprehensive evaluation metrics
    """
    
    console.print(Panel(welcome_text, title="AI Agent Prototype", border_style="green"))


@app.command()
def prepare_data():
    """Prepare training and test datasets"""
    console.print("\n[bold blue]Preparing datasets...[/bold blue]")
    
    preparator = DataPreparator()
    preparator.save_datasets()
    
    console.print("[green]✓ Data preparation complete![/green]")


@app.command()
def train():
    """Fine-tune the model using LoRA"""
    console.print("\n[bold blue]Starting fine-tuning process...[/bold blue]")
    console.print("[yellow]This will take several minutes.[/yellow]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Training model...", total=None)
        
        fine_tuner = FineTuner(base_model_name="distilgpt2")
        fine_tuner.train()
        
        progress.update(task, completed=True)
    
    console.print("[green]✓ Fine-tuning complete![/green]")


@app.command()
def test_summarize(
    text: str = typer.Option("", "--text", "-t", help="Text to summarize"),
    file: str = typer.Option("", "--file", "-f", help="File containing text to summarize"),
    use_finetuned: bool = typer.Option(True, "--use-finetuned/--use-base", help="Use fine-tuned model")
):
    """Test the summarization capabilities"""
    if not text and not file:
        console.print("[red]Error: Provide either --text or --file option[/red]")
        raise typer.Exit(1)
    
    if file:
        file_path = Path(file)
        if not file_path.exists():
            console.print(f"[red]Error: File not found: {file}[/red]")
            raise typer.Exit(1)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    console.print("\n[bold blue]Initializing agent...[/bold blue]")
    agent = AIAgent(use_finetuned=use_finetuned)
    
    console.print("[bold blue]Summarizing text...[/bold blue]\n")
    summary = agent.summarize_academic_paper(text)
    
    console.print(Panel(summary, title="Summary", border_style="cyan"))


@app.command()
def evaluate(
    baseline: bool = typer.Option(False, "--baseline", help="Include baseline comparison")
):
    """Evaluate agent performance"""
    console.print("\n[bold blue]Running evaluation...[/bold blue]")
    
    evaluator = Evaluator()
    test_data = evaluator.load_test_data()
    
    # This is a placeholder - in practice, you'd run the actual agent
    console.print("[yellow]Note: For full evaluation, agent needs to be implemented[/yellow]")
    
    from evaluation import generate_baseline_predictions
    texts = [item['input'] for item in test_data]
    references = [item['output'] for item in test_data]
    predictions = generate_baseline_predictions(texts)
    
    metrics = evaluator.evaluate_generation(predictions, references)
    
    # Display results in table
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta", justify="right")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key.replace('_', ' ').title(), f"{value:.4f}")
    
    console.print("\n")
    console.print(table)
    
    # Save report
    report_path = OUTPUT_DIR / "evaluation_report.json"
    evaluator.generate_evaluation_report(metrics, report_path)
    console.print(f"\n[green]✓ Report saved to: {report_path}[/green]")


@app.command()
def setup_rag():
    """Initialize RAG knowledge base"""
    console.print("\n[bold blue]Setting up RAG system...[/bold blue]")
    
    rag = RAGSystem()
    rag.initialize_knowledge_base()
    
    console.print("[green]✓ RAG system initialized![/green]")


@app.command()
def interactive():
    """Launch interactive mode"""
    console.print("\n[bold green]Interactive Mode[/bold green]")
    console.print("Type 'exit' to quit\n")
    
    agent = AIAgent(use_finetuned=True)
    
    while True:
        try:
            task = console.input("[bold cyan]Your task: [/bold cyan]")
            
            if task.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not task.strip():
                continue
            
            # Optional: collect context if user typed a summarization intent without content
            context = None
            tl = task.lower().strip()
            if tl.startswith("summarize this") or tl.startswith("summarise this"):
                import re
                m = re.match(r"^\s*(summari[sz]e\s+this)\s*[:\-]?\s*(.*)$", task, flags=re.IGNORECASE | re.DOTALL)
                extracted = (m.group(2) or "").strip() if m else ""
                if not extracted:
                    console.print("[dim]Paste the text to summarize (single line is fine):[/dim]")
                    extracted = console.input("[bold cyan]Text: [/bold cyan]")
                if extracted:
                    context = extracted
                    task = "Summarize the provided text concisely"

            # Process task
            with console.status("[bold green]Processing..."):
                result = agent.process_task(task, context)
            
            # Display results
            console.print("\n[bold]Plan:[/bold]")
            for i, step in enumerate(result['plan']['steps'], 1):
                console.print(f"{i}. {step}")
            
            console.print(f"\n[bold]Confidence:[/bold] {result['plan']['confidence']:.2f}")
            
            console.print("\n[bold]Result:[/bold]")
            console.print(Panel(
                result['final_result'][:500] + "..." if len(result['final_result']) > 500 else result['final_result'],
                border_style="green"
            ))
            console.print("\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def status():
    """Check system status and configuration"""
    console.print("\n[bold blue]System Status[/bold blue]\n")
    
    # Check files
    table = Table(title="File Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Path")
    
    from config import (
        TRAINING_DATA_PATH, TEST_DATA_PATH,
        FINE_TUNED_MODEL_PATH, OUTPUT_DIR
    )
    
    components = [
        ("Training Data", TRAINING_DATA_PATH),
        ("Test Data", TEST_DATA_PATH),
        ("Fine-tuned Model", FINE_TUNED_MODEL_PATH),
        ("Output Directory", OUTPUT_DIR)
    ]
    
    for name, path in components:
        exists = Path(path).exists()
        status_text = "[green]✓ Ready[/green]" if exists else "[red]✗ Missing[/red]"
        table.add_row(name, status_text, str(path))
    
    console.print(table)
    
    # Display configuration
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"Model: DistilGPT2 with LoRA")
    console.print(f"Task: Academic Paper Summarization")
    console.print(f"Features: Multi-agent, RAG, Comprehensive Evaluation")


@app.command()
def demo():
    """Run a complete demonstration"""
    console.print("\n[bold green]Running Complete Demonstration[/bold green]\n")
    
    # Step 1: Welcome
    welcome()
    
    # Step 2: Check status
    console.print("\n")
    status()
    
    # Step 3: Prepare data if needed
    if not Path("data/training_data.json").exists():
        console.print("\n")
        prepare_data()
    
    # Step 4: Show example usage
    console.print("\n[bold]Example Usage:[/bold]")
    example_text = """
Researchers have developed a novel approach to quantum computing that uses superconducting qubits. 
The method achieves 99.9% gate fidelity and reduces decoherence times by 50%. 
Experimental validation on IBM quantum hardware demonstrates practical scalability.
"""
    
    console.print(Panel(example_text, title="Example Input", border_style="blue"))
    
    console.print("\n[yellow]To test summarization, run: python cli.py test-summarize --text \"...\"[/yellow]")
    console.print("[yellow]Or start interactive mode: python cli.py interactive[/yellow]\n")


if __name__ == "__main__":
    app()

