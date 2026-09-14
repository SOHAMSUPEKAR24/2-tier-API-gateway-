import requests
import time

def run_tests():
    url = "http://localhost:8000"
    
    # 1. Benign request (small, no sql keywords)
    print("Sending Benign Request...")
    r1 = requests.get(f"{url}/api/v1/users")
    print("Response:", r1.json())
    
    # 2. Block request (sql injection keywords)
    print("\nSending Attack Request (Block expected)...")
    r2 = requests.get(f"{url}/api/v1/users?id=1%20OR%201=1")
    try:
        print("Response:", r2.json())
    except:
        print("Response:", r2.status_code, r2.text)
        
    # 3. Escalate request (upload path)
    print("\nSending Upload Request (Escalation expected)...")
    r3 = requests.post(f"{url}/api/v1/upload", data=b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*")
    print("Response:", r3.json())

    # 4. Trigger feedback loop by sending multiple escalations
    for i in range(11):
        requests.post(f"{url}/api/v1/upload", data=b"dummy")
        
if __name__ == "__main__":
    run_tests()
