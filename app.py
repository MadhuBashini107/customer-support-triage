"""
HuggingFace Spaces entry point.
Redirects to the FastAPI app served on port 7860.
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # On HF Spaces the working dir contains the server files
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
    
    import uvicorn
    from server.main import app
    uvicorn.run(app, host="0.0.0.0", port=7860)
