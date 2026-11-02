"""Interactive Control Center for modeloINV.

Provides a user-friendly terminal menu or direct execution mode
to run the 5 operational steps with custom parameters, live logs,
and automatic browser dashboard launch.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser

BANNER = r"""
========================================================================
             _     _       _____ _   ___     __
  _ __  ___ | | __| |___  |_   _| \ | \ \   / /
 | '_ ` _ \ / _ \/ _` / _ \   | | |  \| |\ \ / / 
 | | | | | | (_) | (_| (_) |  | | | |\  | \ V /  
 |_| |_| |_|\___/ \__,_\___/ |___|_| \_|  \_/   
 Micro-Capital €100 Quantitative Trading System
========================================================================
"""

MENU = """
Choose an action to run:

  [1] 📊 Run Historical Backtest (12 Months, €100 initial capital)
  [2] 🖥️  Launch Real-Time Web Dashboard (FastAPI + Browser UI)
  [3] 📈 Start Paper-Trading Simulation Engine (Live Intraday Loop)
  [4] ⚡ Trigger Continuous Learning Retraining (Model Registry)
  [5] 🚦 Evaluate 8-Gate Live Readiness (Check if ready for real €100)

  [6] 🧪 Run Full Test Suite (pytest - 567 tests)
  [0] 🚪 Exit

"""

def run_command(cmd_args: list[str], title: str, wait: bool = True):
    print("\n" + "=" * 72)
    print(f"  EXECUTING: {title}")
    print(f"  COMMAND  : {' '.join(cmd_args)}")
    print("=" * 72 + "\n")
    try:
        proc = subprocess.Popen(cmd_args, env=os.environ.copy())
        if wait:
            proc.wait()
            print("\n" + "-" * 72)
            print(f"  [DONE] {title} finished with return code: {proc.returncode}")
            print("-" * 72 + "\n")
        return proc
    except Exception as exc:
        print(f"  [ERROR] Failed to run {title}: {exc}")
        return None

def step_backtest():
    print("\n-- Step 1: Historical Backtest Simulation --")
    months = input("Simulation duration in months [default: 12]: ").strip() or "12"
    capital = input("Starting capital in EUR [default: 100.0]: ").strip() or "100.0"
    persist = input("Persist results to local database? (y/n) [default: y]: ").strip().lower() != "n"
    
    cmd = [sys.executable, "-m", "modeloinv.cli", "backtest", "--months", months, "--capital", capital]
    if persist:
        cmd.append("--persist")
    run_command(cmd, "Historical Backtest")

def step_dashboard():
    print("\n-- Step 2: Real-Time Web Dashboard --")
    port = input("Server port [default: 8000]: ").strip() or "8000"
    url = f"http://localhost:{port}"
    print(f"\nLaunching dashboard on {url} ...")
    
    cmd = [sys.executable, "-m", "modeloinv.cli", "dashboard", "--port", port]
    proc = run_command(cmd, "Web Dashboard Server", wait=False)
    
    # Give server a moment to start and launch default browser
    time.sleep(1.5)
    webbrowser.open(url)
    
    print("\n>>> Dashboard is running! Press Ctrl+C in this terminal when you want to stop it. <<<\n")
    try:
        if proc:
            proc.wait()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        if proc:
            proc.terminate()
            proc.wait()
        print("Dashboard stopped.")

def step_paper():
    print("\n-- Step 3: Paper Trading Simulation Engine --")
    capital = input("Starting capital in EUR [default: 100.0]: ").strip() or "100.0"
    print("\nStarting paper trading loop... (Press Ctrl+C to stop)")
    cmd = [sys.executable, "-m", "modeloinv.cli", "paper", "--capital", capital]
    proc = run_command(cmd, "Paper Trading Loop", wait=False)
    try:
        if proc:
            proc.wait()
    except KeyboardInterrupt:
        print("\nStopping paper trading loop...")
        if proc:
            proc.terminate()
            proc.wait()
        print("Paper trading stopped.")

def step_retrain():
    print("\n-- Step 4: Continuous Learning Retraining --")
    force = input("Force retrain bypassing cooldown? (y/n) [default: y]: ").strip().lower() != "n"
    cmd = [sys.executable, "-m", "modeloinv.cli", "retrain"]
    if force:
        cmd.append("--force")
    run_command(cmd, "Continuous Learning Retraining")

def step_readiness():
    print("\n-- Step 5: 8-Gate Live Readiness Evaluation --")
    print("Checking whether the model is statistically certified for real money...")
    cmd = [sys.executable, "-m", "modeloinv.cli", "evaluate", "--simulate"]
    run_command(cmd, "8-Gate Live Readiness Audit")

def step_tests():
    print("\n-- Run Test Suite --")
    cmd = [sys.executable, "-m", "pytest", "-q"]
    run_command(cmd, "Pytest Test Suite (567 Tests)")

def main():
    while True:
        print(BANNER)
        print(MENU)
        try:
            choice = input("Enter option [0-6]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if choice == "1":
            step_backtest()
        elif choice == "2":
            step_dashboard()
        elif choice == "3":
            step_paper()
        elif choice == "4":
            step_retrain()
        elif choice == "5":
            step_readiness()
        elif choice == "6":
            step_tests()
        elif choice == "0":
            print("\nExiting modeloINV Control Center. Good luck trading!\n")
            break
        else:
            print(f"\n[!] Invalid option: '{choice}'. Please select a number from 0 to 6.")
        
        input("\nPress Enter to return to main menu...")

if __name__ == "__main__":
    main()
