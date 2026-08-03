import subprocess
import json
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def call_mcp_tool(tool_name, arguments={}):
    command_args = ["node", "D:\\SocratiCode-main\\dist\\index.js"]
    
    try:
        process = subprocess.Popen(
            command_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding='utf-8'
        )
        
        # 1. Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-client", "version": "1.0.0"}
            }
        }
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        process.stdout.readline() # Consume initialize response
        
        # 2. Call Tool
        call_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        process.stdin.write(json.dumps(call_request) + "\n")
        process.stdin.flush()
        
        response_str = process.stdout.readline()
        process.terminate()
        
        if response_str:
            res = json.loads(response_str)
            if "error" in res:
                return f"Error: {res['error']}"
            
            # Extract content text from MCP response
            content = res.get("result", {}).get("content", [])
            output = []
            for item in content:
                if item.get("type") == "text":
                    output.append(item.get("text"))
            return "\n".join(output)
        return "No response from server"
    except Exception as e:
        return f"Exception: {str(e)}"

if __name__ == "__main__":
    print("=== SOCRATICODE HEALTH CHECK ===", flush=True)
    health = call_mcp_tool("codebase_health")
    print(health, flush=True)
    
    print("\n=== SOCRATICODE PROJECTS ===", flush=True)
    projects = call_mcp_tool("codebase_list_projects")
    print(projects, flush=True)
