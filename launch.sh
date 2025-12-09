pip install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate

uv pip install -e .
uv cache prune

source .venv/bin/activate