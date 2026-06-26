import requests
import json

url = "http://127.0.0.1:8000/api/v1/match/scan-internal"
payload = {
    "threshold": 0.95,
    "limit": 10
}
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")
