"""
Customer Support Agent

An AI agent specialized for handling customer support queries.
"""

from typing import Optional
from .base_agent import BaseAgent, Tool
from ..llm.providers import LLMProvider


class CustomerSupportAgent(BaseAgent):
    """
    Customer Support Agent.
    
    Handles:
    - Product/service inquiries
    - Order status and tracking
    - Returns and refunds
    - General business questions
    """
    
    @property
    def name(self) -> str:
        return "Customer Support Agent"
    
    @property
    def system_prompt(self) -> str:
        return """You are a professional customer support agent.

PERSONALITY:
- Warm, helpful, and patient
- Professional but approachable
- Uses respectful forms of address

CAPABILITIES:
- Answer product and service questions
- Help with order status inquiries
- Assist with returns and refunds process
- Guide through common business processes

GUIDELINES:
1. Always greet politely
2. Ask clarifying questions if the query is unclear
3. Be honest when you don't know something
4. Offer to escalate to a human when appropriate
5. Confirm understanding before providing solutions

LIMITATIONS:
- Cannot process actual transactions
- Cannot access real-time order databases
- Cannot provide legal or financial advice
- For complex issues, recommend speaking with a human agent

ESCALATION TRIGGERS (suggest human agent):
- Complaints about serious issues
- Legal disputes
- Financial discrepancies
- Requests for manager/supervisor
- Emotional distress

Always end with asking if there's anything else you can help with."""
    
    def _register_tools(self):
        """Register customer support tools."""
        
        # Order Status (mock)
        self.add_tool(Tool(
            name="check_order_status",
            description="Check the status of an order by order ID",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up"
                    }
                },
                "required": ["order_id"]
            },
            function=self._check_order_status
        ))
        
        # Calculate Invoice
        self.add_tool(Tool(
            name="calculate_invoice",
            description="Calculate an invoice with tax",
            parameters={
                "type": "object",
                "properties": {
                    "base_amount": {
                        "type": "number",
                        "description": "Base amount"
                    },
                    "tax_rate": {
                        "type": "number",
                        "description": "Tax rate as percentage (e.g., 10 for 10%)"
                    }
                },
                "required": ["base_amount", "tax_rate"]
            },
            function=self._calculate_invoice
        ))
        
        # Escalate to Human
        self.add_tool(Tool(
            name="escalate_to_human",
            description="Escalate the issue to a human support agent",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for escalation"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "description": "Priority level"
                    }
                },
                "required": ["reason", "priority"]
            },
            function=self._escalate_to_human
        ))
    
    def _check_order_status(self, order_id: str) -> str:
        """Check order status (mock implementation)."""
        mock_statuses = {
            "default": "Order received and being processed",
            "ORD": "Order confirmed, preparing for shipment",
            "SHP": "Shipped via courier, in transit",
            "DLV": "Delivered successfully",
        }
        
        prefix = order_id[:3].upper() if len(order_id) >= 3 else "default"
        status = mock_statuses.get(prefix, mock_statuses["default"])
        
        return f"""Order Status for {order_id}:
Status: {status}

Note: This is a demo system. In production, this would show real order tracking."""
    
    def _calculate_invoice(
        self, 
        base_amount: float, 
        tax_rate: float
    ) -> str:
        """Calculate invoice with tax breakdown."""
        tax_amount = base_amount * (tax_rate / 100)
        total = base_amount + tax_amount
        
        return f"""Invoice Calculation:
─────────────────────────
Base Amount: ${base_amount:,.2f}
Tax ({tax_rate}%): ${tax_amount:,.2f}
─────────────────────────
TOTAL: ${total:,.2f}
═════════════════════════"""
    
    def _escalate_to_human(self, reason: str, priority: str) -> str:
        """Create escalation ticket (mock)."""
        ticket_id = f"ESC-{hash(reason) % 10000:04d}"
        
        priority_times = {
            "urgent": "within 1 hour",
            "high": "within 4 hours",
            "medium": "within 24 hours",
            "low": "within 48 hours"
        }
        
        return f"""Escalation Created:
─────────────────────────
Ticket ID: {ticket_id}
Priority: {priority.upper()}
Reason: {reason}
Expected Response: {priority_times.get(priority, 'within 24 hours')}
─────────────────────────

A human support agent will contact you soon."""


def create_customer_support_agent(
    provider: LLMProvider = LLMProvider.NVIDIA,
    api_key: Optional[str] = None
) -> CustomerSupportAgent:
    """Factory function to create a customer support agent."""
    return CustomerSupportAgent(provider=provider, api_key=api_key)
