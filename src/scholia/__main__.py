"""Allow `python -m scholia` to run the CLI (parity with the `scholia` script)."""

from scholia.cli import cli

if __name__ == "__main__":  # pragma: no cover
    cli()
