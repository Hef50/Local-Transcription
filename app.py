"""GUI entry point (double-click run.bat, or: .venv\\Scripts\\pythonw.exe app.py).

`app.py --worker ...` runs the headless pipeline instead. The GUI launches itself
that way for each job, which also keeps a future single-file .exe working.
"""
import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        from recpipe.cli import main as cli_main

        sys.exit(cli_main(sys.argv[2:]))
    from recpipe.gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
