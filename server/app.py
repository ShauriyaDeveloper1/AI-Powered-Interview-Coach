"""Server entrypoint required by multi-mode deployment validators."""

import os

from app import app


def main() -> None:
    """Run the Flask app using the expected server entrypoint."""
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
