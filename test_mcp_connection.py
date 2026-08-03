import subprocess
import json
import time
import sys

def log(msg):
    print(msg, flush=True)
    with open("mcp_test_results.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear the results file
with open("mcp_test_results.txt", "w", encoding="utf-8") as f:
    f.write("=== MCP CONNECTION TEST RESULTS ===\n")

def test_mcp_server(name, command_args):
    log(f"\nTesting MCP Server: {name}")
    log(f"Command: {' '.join(command_args)}")
    
    try:
        # Start the MCP server process
        process = subprocess.Popen(
            command_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding='utf-8'
        )
        
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "mcp-test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        log("Sending initialize request...")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Wait a short bit to read
        log("Waiting for response...")
        time.sleep(2)
        
        # Read from stdout using non-blocking or simple read
        # Since we don't want to hang, we can check if there's output
        # On Windows, select doesn't work on pipes, but we can do a simple check
        # We can try to read with a timeout using communicate or similar
        # But communicate closes the stdin/stdout. Since initialize is first, we can do communicate
        # with a short timeout!
        try:
            stdout, stderr = process.communicate(input=json.dumps(init_request) + "\n", timeout=5)
            log("Process finished communicating.")
            if stdout:
                lines = stdout.strip().split('\n')
                for line in lines:
                    try:
                        res = json.loads(line)
                        log(f"[OK] Got response: {json.dumps(res)[:200]}...")
                        # If we got initialization, connection is verified!
                        process.terminate()
                        return True
                    except json.JSONDecodeError:
                        continue
            if stderr:
                log(f"Stderr output:\n{stderr}")
        except subprocess.TimeoutExpired:
            log("Timeout expired waiting for response.")
            process.kill()
            stdout, stderr = process.communicate()
            if stdout:
                log(f"Stdout after timeout: {stdout[:500]}")
            if stderr:
                log(f"Stderr after timeout: {stderr[:500]}")
            
        process.terminate()
        return False
        
    except Exception as e:
        log(f"[ERROR] Exception during test: {str(e)}")
        return False

if __name__ == "__main__":
    # 1. Test Postgres MCP
    postgres_cmd = [
        "npx.cmd", "-y", "--prefer-offline", 
        "@modelcontextprotocol/server-postgres", 
        "postgresql://postgres:postgres@127.0.0.1:54325/postgres"
    ]
    test_mcp_server("postgres", postgres_cmd)
    
    # 2. Test Socraticode MCP
    socraticode_cmd = [
        "node", "D:\\SocratiCode-main\\dist\\index.js"
    ]
    test_mcp_server("socraticode", socraticode_cmd)
    
    # 3. Test Puppeteer MCP
    puppeteer_cmd = [
        "npx.cmd", "-y", "--prefer-offline",
        "@modelcontextprotocol/server-puppeteer"
    ]
    test_mcp_server("puppeteer", puppeteer_cmd)
    
    log("\n=== TEST COMPLETED ===")
