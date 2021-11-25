set -euo pipefail

pip install jupyterlab
export PYTHONPATH=/torch_agents
jupyter lab --allow-root --ip 0.0.0.0 -p 8888
