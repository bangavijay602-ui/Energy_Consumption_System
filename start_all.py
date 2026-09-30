"""Full-Stack Startup Script: Launches FastAPI Backend (Port 8000) and React Frontend (Port 5173)."""

import os
import subprocess
import sys
import time

def start_full_stack():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(base_dir, "frontend")

    print("=" * 70)
    print("  VOLTCAST AI — STARTING FULL-STACK APPLICATION")
    print("=" * 70)
    print("1. Launching FastAPI Backend on http://localhost:8000 ...")
    print("2. Launching React Vite Frontend on http://localhost:5173 ...\n")

    # Start FastAPI Backend
    backend_cmd = [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    backend_process = subprocess.Popen(backend_cmd, cwd=base_dir)

    time.sleep(2)

    # Start Vite Frontend
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_cmd = [npm_cmd, "run", "dev"]
    frontend_process = subprocess.Popen(frontend_cmd, cwd=frontend_dir)

    print("\n✅ System Online!")
    print("👉 Access the React Web UI: http://localhost:5173")
    print("👉 Access the FastAPI Docs:   http://localhost:8000/docs\n")
    print("Press Ctrl+C to terminate both servers.")

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Done.")

if __name__ == "__main__":
    start_full_stack()
