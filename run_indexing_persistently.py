import subprocess
import json
import time
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_persistent_indexer():
    command_args = ["node", "D:\\SocratiCode-main\\dist\\index.js"]
    
    print("Starting Socraticode process...", flush=True)
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
    print("Sending initialize handshake...", flush=True)
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "persistent-mcp-client", "version": "1.0.0"}
        }
    }
    process.stdin.write(json.dumps(init_request) + "\n")
    process.stdin.flush()
    init_response = process.stdout.readline()
    print(f"Handshake complete: {init_response[:200]}...", flush=True)
    
    # 2. Trigger Indexing
    print("Sending codebase_index tool call...", flush=True)
    index_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "codebase_index",
            "arguments": {"projectPath": "D:\\product_checker\\check-products"}
        }
    }
    process.stdin.write(json.dumps(index_request) + "\n")
    process.stdin.flush()
    index_response_str = process.stdout.readline()
    
    try:
        index_response = json.loads(index_response_str)
        content_items = index_response.get("result", {}).get("content", [])
        text_content = "\n".join([item.get("text", "") for item in content_items if item.get("type") == "text"])
        print("codebase_index trigger response:", flush=True)
        print(text_content, flush=True)
    except Exception as e:
        print(f"Error parsing trigger response: {e}", flush=True)
        print(f"Raw: {index_response_str}", flush=True)
    
    # 3. Poll status on the SAME process until 100% complete
    print("Starting status polling loop...", flush=True)
    request_id = 3
    in_progress_seen = False
    
    # Run loop for up to 1000 iterations (10,000 seconds = ~2.7 hours)
    for i in range(1000):
        time.sleep(10)
        
        status_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": "codebase_status",
                "arguments": {"projectPath": "D:\\product_checker\\check-products"}
            }
        }
        request_id += 1
        
        try:
            process.stdin.write(json.dumps(status_request) + "\n")
            process.stdin.flush()
            status_response_str = process.stdout.readline()
            
            res = json.loads(status_response_str)
            content_items = res.get("result", {}).get("content", [])
            status_text = "\n".join([item.get("text", "") for item in content_items if item.get("type") == "text"])
            
            # Print status to screen
            print(f"\n--- POLLING STATUS #{i+1} ---", flush=True)
            print(status_text, flush=True)
            
            # Check if indexing is in progress
            is_in_progress = "in progress" in status_text.lower() or "preparing" in status_text.lower() or "indexing" in status_text.lower()
            
            if is_in_progress:
                in_progress_seen = True
            elif in_progress_seen:
                # If we saw it in progress before and now it's not, it must have finished!
                print("\n[SUCCESS] Indexing is no longer in progress! Exiting loop.", flush=True)
                break
            else:
                # If we haven't seen it in progress yet (e.g. at the very start), check if pointsCount is populated
                # indicating it might have finished instantly because of cache.
                if "Indexed chunks:" in status_text and not is_in_progress:
                    print("\n[SUCCESS] Index is ready and healthy! Exiting loop.", flush=True)
                    break
                    
        except Exception as e:
            print(f"Error in polling loop: {e}", flush=True)
            break
            
    # Terminate process cleanly
    print("Terminating Socraticode process...", flush=True)
    process.terminate()
    print("Done!", flush=True)

if __name__ == "__main__":
    run_persistent_indexer()
