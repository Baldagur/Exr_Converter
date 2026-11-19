import sys


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        from .selftest import run

        return run(args[1] if len(args) > 1 else "")
    if args and args[0] in ("--version", "-v"):
        from . import __version__

        print(__version__)
        return 0
    from .ui.main_window import run_gui

    return run_gui(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
