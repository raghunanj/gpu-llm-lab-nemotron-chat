"""
Command-Line Interface for AI Assistant

A terminal interface for interacting with AI agents.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
import typer

from src.agents import CustomerSupportAgent, CodingAssistantAgent
from src.llm.providers import LLMProvider

# Initialize rich console
console = Console()

# Initialize typer app
app = typer.Typer(
    name="ai-assistant",
    help="🤖 AI Assistant - Powered by NVIDIA Nemotron 3"
)


def print_banner():
    """Print the application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🤖 AI Assistant                                             ║
║   ─────────────────────────────────────────────────────────   ║
║   Powered by NVIDIA Nemotron 3                                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner.strip(), style="bold blue"))


def print_help():
    """Print available commands."""
    table = Table(title="Available Commands", show_header=True)
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    
    table.add_row("help", "Show this help message")
    table.add_row("switch", "Switch between agents (support/coding)")
    table.add_row("info", "Show current agent info")
    table.add_row("reset", "Reset conversation history")
    table.add_row("clear", "Clear the screen")
    table.add_row("quit/exit", "Exit the application")
    
    console.print(table)


def get_agent(agent_type: str):
    """Get the appropriate agent instance."""
    if agent_type == "support":
        return CustomerSupportAgent()
    elif agent_type == "coding":
        return CodingAssistantAgent()
    else:
        return CustomerSupportAgent()


def interactive_chat(agent_type: str = "support"):
    """Run interactive chat session."""
    agent = get_agent(agent_type)
    
    console.print(f"\n✅ {agent.name} is ready!", style="bold green")
    console.print("Type 'help' for commands, 'quit' to exit.\n")
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["quit", "exit", "bye"]:
                console.print("\n👋 Goodbye!", style="bold blue")
                break
            
            elif user_input.lower() == "help":
                print_help()
                continue
            
            elif user_input.lower() == "reset":
                agent.reset()
                console.print("🔄 Conversation reset.", style="yellow")
                continue
            
            elif user_input.lower() == "clear":
                console.clear()
                print_banner()
                continue
            
            elif user_input.lower() == "info":
                info = agent.get_info()
                console.print(Panel(
                    f"Agent: {info['name']}\n"
                    f"Tools: {', '.join(info['tools']) if info['tools'] else 'None'}\n"
                    f"Memory: {info['memory_size']} messages",
                    title="Agent Info"
                ))
                continue
            
            elif user_input.lower() == "switch":
                new_type = Prompt.ask(
                    "Switch to", 
                    choices=["support", "coding"],
                    default="support"
                )
                agent = get_agent(new_type)
                console.print(f"✅ Switched to {agent.name}", style="bold green")
                continue
            
            # Get AI response
            console.print()
            with console.status("[bold green]Thinking...", spinner="dots"):
                response = agent.chat(user_input)
            
            # Display response
            console.print(f"[bold green]🤖 {agent.name}[/bold green]")
            console.print(Markdown(response))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n\n👋 Goodbye!", style="bold blue")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def chat(
    agent: str = typer.Option(
        "support", 
        "--agent", "-a",
        help="Agent type: support, coding"
    )
):
    """Start an interactive chat session with an AI agent."""
    print_banner()
    interactive_chat(agent)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question"),
    agent: str = typer.Option("support", "--agent", "-a", help="Agent type")
):
    """Ask a single question and get an answer."""
    ai_agent = get_agent(agent)
    
    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")
    
    with console.status("[bold green]Thinking...", spinner="dots"):
        response = ai_agent.chat(question)
    
    console.print(f"[bold green]Answer:[/bold green]")
    console.print(Markdown(response))


@app.command()
def agents():
    """List available agents."""
    table = Table(title="Available Agents", show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Description", style="dim")
    
    table.add_row(
        "support",
        "Customer Support Agent",
        "Handle customer queries and support requests"
    )
    table.add_row(
        "coding",
        "Coding Assistant",
        "Code review, debugging, documentation"
    )
    
    console.print(table)


def main():
    """Main entry point."""
    if len(sys.argv) == 1:
        # No arguments - start interactive chat
        print_banner()
        
        # Select agent
        console.print("\n[bold]Select an agent:[/bold]")
        console.print("1. Customer Support Agent")
        console.print("2. Coding Assistant")
        
        choice = Prompt.ask(
            "Your choice",
            choices=["1", "2"],
            default="1"
        )
        
        agent_type = "support" if choice == "1" else "coding"
        interactive_chat(agent_type)
    else:
        app()


if __name__ == "__main__":
    main()
