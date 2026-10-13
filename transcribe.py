"""CLI entry point:  python transcribe.py recording.m4a --model medium.en"""
import sys

from recpipe.cli import main

if __name__ == "__main__":
    sys.exit(main())
