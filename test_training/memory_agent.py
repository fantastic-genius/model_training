# memory_agent.py
import ollama
import json
from datetime import datetime

class MemoryAgent:
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.long_term_memory = []  # Facts to remember
        self.conversation_history = []
        
    def remember(self, fact: str):
        """Store a fact in long-term memory"""
        self.long_term_memory.append({
            'fact': fact,
            'timestamp': datetime.now().isoformat()
        })
        
    def recall(self) -> str:
        """Retrieve all memories"""
        if not self.long_term_memory:
            return "No memories stored."
        return "\n".join([f"- {m['fact']}" for m in self.long_term_memory])
    
    def chat(self, message: str) -> str:
        """Chat with memory context"""
        # Build context with memories
        system_prompt = f"""You are a helpful assistant with memory.

            Your memories:
            {self.recall()}

            Use these memories to personalize your responses."""

        self.conversation_history.append({
            'role': 'user',
            'content': message
        })
        
        response = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt}
            ] + self.conversation_history[-10:]  # Last 10 messages
        )
        
        assistant_message = response['message']['content']
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': assistant_message
        })
        
        # Auto-extract facts to remember
        if any(keyword in message.lower() for keyword in ['my name is', 'i am', 'i work', 'i like']):
            self.remember(message)
        
        return assistant_message
    
    def save_memory(self, filepath: str = "agent_memory.json"):
        """Persist memory to disk"""
        with open(filepath, 'w') as f:
            json.dump({
                'memories': self.long_term_memory,
                'history': self.conversation_history
            }, f, indent=2)
    
    def load_memory(self, filepath: str = "agent_memory.json"):
        """Load memory from disk"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.long_term_memory = data.get('memories', [])
                self.conversation_history = data.get('history', [])
        except FileNotFoundError:
            print("No saved memory found")

# Usage
agent = MemoryAgent()

print(agent.chat("My name is Hamzah and I lead engineering at SameDayCustom"))
print("\n" + "="*50 + "\n")
print(agent.chat("What's my name?"))
print("\n" + "="*50 + "\n")
print(agent.chat("What do I do for work?"))

# Save for next session
agent.save_memory()