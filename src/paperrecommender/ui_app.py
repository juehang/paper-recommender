#!/usr/bin/env python3
import argparse
import os
import sys
from .ui import start_app

def main():
    """
    Main entry point for the Paper Recommender UI application.
    """
    parser = argparse.ArgumentParser(description="Paper Recommender UI")
    
    # UI options
    parser.add_argument("--mode", default="chrome", choices=["chrome", "electron", "browser", "default"],
                        help="Mode to start Eel in (default: chrome)")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--no-block", action="store_true", help="Don't block the main thread")
    
    args = parser.parse_args()
    
    # Start the app
    try:
        start_app(
            mode=args.mode,
            host=args.host,
            port=args.port,
            block=not args.no_block
        )
    except Exception as e:
        print(f"Error starting UI: {e}", file=sys.stderr)
        
        # If chrome mode fails, try browser mode
        if args.mode == "chrome":
            print("Trying browser mode instead...", file=sys.stderr)
            try:
                start_app(
                    mode="browser",
                    host=args.host,
                    port=args.port,
                    block=not args.no_block
                )
            except Exception as e2:
                print(f"Error starting UI in browser mode: {e2}", file=sys.stderr)
                sys.exit(1)
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()
