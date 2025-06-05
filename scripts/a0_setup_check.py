# scripts/a0_setup_check.py
import ee
import os
import certifi

# Override problematic certificate environment variables.
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()
os.environ.pop("REQUESTS_CA_BUNDLE", None)

def setup_check():
    print("Initializing Google Earth Engine API...")
    try:
        ee.Initialize(project="earthenginecapeverde")
        print("Earth Engine API initialized successfully!")
    except Exception as e:
        print("ERROR: Failed to initialize Earth Engine API. Check credentials & internet.")
        print("Details:", e)
        exit(1)

if __name__ == "__main__":
    setup_check()
