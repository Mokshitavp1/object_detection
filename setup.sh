#!/bin/sh
set -e

PY=$(command -v python3)
if [ -z "$PY" ]; then
    echo "Error: python3 not found."
    exit 1
fi

MAJOR=$($PY -c "import sys; print(sys.version_info.major)")
MINOR=$($PY -c "import sys; print(sys.version_info.minor)")

if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    echo "Error: Python 3.10+ required. Found $MAJOR.$MINOR."
    exit 1
else
    echo "Python $MAJOR.$MINOR found."
fi

$PY -m venv .venv
echo "Virtualenv created at .venv/"

.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt
echo "Dependencies installed."

echo ""
echo "Setup complete."
echo "Run:"
echo "  source .venv/bin/activate"
echo "  uvicorn server:app --reload"
