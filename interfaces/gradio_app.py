"""
Gradio Web Interface for AI Assistant

A modern web UI for interacting with AI agents.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from typing import Generator

from src.agents import CustomerSupportAgent, CodingAssistantAgent
from src.llm.providers import LLMProvider


# Initialize agents
agents = {
    "Customer Support": CustomerSupportAgent(),
    "Coding Assistant": CodingAssistantAgent(),
}

current_agent_name = "Customer Support"


def get_current_agent():
    """Get the currently selected agent."""
    return agents[current_agent_name]


def switch_agent(agent_name: str):
    """Switch to a different agent."""
    global current_agent_name
    current_agent_name = agent_name
    agent = agents[agent_name]
    agent.reset()
    return f"Switched to {agent.name}. Conversation reset."


def chat_response(message: str, history: list) -> str:
    """Process a chat message and return the response."""
    agent = get_current_agent()
    response = agent.chat(message)
    return response


def chat_stream(message: str, history: list) -> Generator[str, None, None]:
    """Stream a chat response."""
    agent = get_current_agent()
    
    full_response = ""
    try:
        for chunk in agent.stream_chat(message):
            full_response += chunk
            yield full_response
    except Exception as e:
        yield f"Error: {str(e)}"


def reset_conversation():
    """Reset the current agent's conversation."""
    agent = get_current_agent()
    agent.reset()
    return []


def get_agent_info():
    """Get information about the current agent."""
    agent = get_current_agent()
    info = agent.get_info()
    
    tools_list = "\n".join(f"• {t}" for t in info['tools']) if info['tools'] else "None"
    
    return f"""**{info['name']}**

**Tools Available:**
{tools_list}

**Memory:** {info['memory_size']} messages
"""


# Custom CSS for a modern look
custom_css = """
:root {
    --primary-color: #6366f1;
    --secondary-color: #1e293b;
    --accent-color: #22d3ee;
    --bg-color: #f8fafc;
    --text-color: #1e293b;
}

.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    max-width: 1200px !important;
    margin: auto !important;
}

.main-header {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, var(--secondary-color) 0%, #475569 100%);
    border-radius: 12px;
    margin-bottom: 1rem;
}

.main-header h1 {
    color: white;
    font-size: 2.5rem;
    margin: 0;
}

.main-header p {
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

.chat-container {
    border: 2px solid var(--accent-color) !important;
    border-radius: 12px !important;
}

.info-panel {
    background: rgba(99, 102, 241, 0.1);
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid var(--primary-color);
}

footer {
    text-align: center;
    padding: 1rem;
    color: #666;
    font-size: 0.9rem;
}
"""

# Example queries
support_examples = [
    "What are your business hours?",
    "I need help with my order",
    "How do I track my shipment?",
    "Can I get a refund?",
    "What payment methods do you accept?",
]

coding_examples = [
    "Explain how async/await works in Python",
    "How do I connect to a PostgreSQL database?",
    "Write a function to validate email addresses",
    "What's the best way to handle errors in FastAPI?",
    "Review this code: def add(a, b): return a+b",
]


def get_examples():
    """Get examples for the current agent."""
    if current_agent_name == "Customer Support":
        return support_examples
    return coding_examples


# Build the Gradio interface
with gr.Blocks(css=custom_css, title="AI Assistant") as demo:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🤖 AI Assistant</h1>
            <p>Powered by NVIDIA Nemotron 3</p>
        </div>
    """)
    
    with gr.Row():
        # Main chat area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_copy_button=True,
                bubble_full_width=False,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here... (Press Enter to send)",
                    label="Your Message",
                    scale=4,
                    autofocus=True,
                )
                send_btn = gr.Button("Send 📤", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                reset_btn = gr.Button("🔄 Reset Agent", variant="secondary")
        
        # Sidebar
        with gr.Column(scale=1):
            # Agent Selection
            gr.Markdown("### 🎯 Select Agent")
            agent_dropdown = gr.Dropdown(
                choices=list(agents.keys()),
                value="Customer Support",
                label="Active Agent",
                interactive=True,
            )
            switch_status = gr.Textbox(
                label="Status",
                value="Customer Support Agent active",
                interactive=False,
            )
            
            gr.Markdown("---")
            
            # Agent Info
            gr.Markdown("### ℹ️ Agent Info")
            agent_info = gr.Markdown(get_agent_info())
            
            gr.Markdown("---")
            
            # Example Queries
            gr.Markdown("### 💡 Try These")
            
            @gr.render(inputs=agent_dropdown)
            def show_examples(agent):
                examples = support_examples if agent == "Customer Support" else coding_examples
                for ex in examples[:4]:
                    btn = gr.Button(ex[:40] + "..." if len(ex) > 40 else ex, size="sm")
                    btn.click(
                        fn=lambda x=ex: x,
                        outputs=msg
                    )
    
    # Footer
    gr.HTML("""
        <footer>
            <p>Powered by NVIDIA Nemotron 3</p>
        </footer>
    """)
    
    # Event handlers
    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        
        bot_response = chat_response(message, chat_history)
        chat_history.append((message, bot_response))
        return "", chat_history
    
    def on_switch(agent_name):
        status = switch_agent(agent_name)
        info = get_agent_info()
        return status, info, []
    
    # Connect events
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot], [msg, chatbot])
    
    clear_btn.click(lambda: [], outputs=chatbot)
    reset_btn.click(
        fn=lambda: (reset_conversation(), get_agent_info()),
        outputs=[chatbot, agent_info]
    )
    
    agent_dropdown.change(
        fn=on_switch,
        inputs=agent_dropdown,
        outputs=[switch_status, agent_info, chatbot]
    )


def main():
    """Launch the Gradio app."""
    print("🚀 Starting AI Assistant...")
    print("=" * 50)
    print("Open your browser at: http://localhost:7860")
    print("=" * 50)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
