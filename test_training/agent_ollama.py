# agent_ollama.py
import ollama
import subprocess
import json
from typing import Dict, List, Callable

class SimpleAgent:
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.conversation_history = []
        self.tools = {}
    
    def register_tool(self, name: str, func: Callable, description: str):
        """Register a tool the agent can use"""
        self.tools[name] = {
            'func': func,
            'description': description
        }
    
    def _create_system_prompt(self) -> str:
        tools_desc = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
        
        return f"""You are a helpful AI agent with access to these tools:

{tools_desc}

When you need to use a tool, respond ONLY with a JSON object in this format:
{{"tool": "tool_name", "input": "your input here"}}

When you have the final answer, respond normally without JSON."""
    
    def run(self, user_input: str, max_iterations: int = 5) -> str:
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        
        for iteration in range(max_iterations):
            # Get response from model
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self._create_system_prompt()}
                ] + self.conversation_history
            )
            
            assistant_message = response['message']['content']
            print(f"\n[Iteration {iteration + 1}] Assistant: {assistant_message}")
            
            # Check if it's a tool call
            if assistant_message.strip().startswith('{') and '"tool"' in assistant_message:
                try:
                    tool_call = json.loads(assistant_message)
                    tool_name = tool_call['tool']
                    tool_input = tool_call['input']
                    
                    # Execute tool
                    if tool_name in self.tools:
                        result = self.tools[tool_name]['func'](tool_input)
                        print(f"[Tool Result] {tool_name}: {result}")
                        
                        # Add tool result to conversation
                        self.conversation_history.append({
                            'role': 'assistant',
                            'content': assistant_message
                        })
                        self.conversation_history.append({
                            'role': 'user',
                            'content': f"Tool result: {result}"
                        })
                    else:
                        return f"Error: Unknown tool '{tool_name}'"
                        
                except json.JSONDecodeError:
                    # Not valid JSON, treat as final answer
                    return assistant_message
            else:
                # Final answer
                return assistant_message
        
        return "Max iterations reached without final answer"

# Define tools
def search_files(query: str) -> str:
    """Search for files on Mac using Spotlight"""
    try:
        result = subprocess.run(
            ['mdfind', '-name', query],
            capture_output=True,
            text=True,
            timeout=5
        )
        files = result.stdout.strip().split('\n')[:5]  # Top 5 results
        return '\n'.join(files) if files[0] else "No files found"
    except Exception as e:
        return f"Error: {str(e)}"

def execute_python(code: str) -> str:
    """Execute Python code safely"""
    try:
        # Create a restricted environment
        allowed_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'sum': sum,
                'max': max,
                'min': min,
            }
        }
        local_vars = {}
        exec(code, allowed_globals, local_vars)
        return str(local_vars.get('result', 'Code executed successfully'))
    except Exception as e:
        return f"Error: {str(e)}"

def read_file(filepath: str) -> str:
    """Read contents of a file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()[:1000]  # First 1000 chars
        return content
    except Exception as e:
        return f"Error: {str(e)}"

# Create and use agent
agent = SimpleAgent(model="llama3.2:3b")

agent.register_tool(
    "search_files",
    search_files,
    "Search for files by name on the computer"
)

agent.register_tool(
    "execute_python",
    execute_python,
    "Execute Python code. Set result variable for output."
)

agent.register_tool(
    "read_file",
    read_file,
    "Read the contents of a file"
)

# Test it
result = agent.run("Find all image files in my desktop folder where the root is /Users/hamzahabdulfattah and count them")
print(f"\n[Final Answer] {result}")

# /Users/hamzahabdulfattah/Documents/PythonProject/model_training/test_training/fine_tunning_2.py