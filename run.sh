#!/usr/bin/env bash

# This script acts as a command "menu" for this cogment project.
# - You can list the available commands using `./run.sh commands`
# - You can add a command as a bash function in this file

set -o errexit

ROOT_DIR=$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_SCRIPT="./$(basename -- "${BASH_SOURCE[0]}")"

COGMENT_VERSION=2.2.0
COGMENT_DIR=${ROOT_DIR}/.cogment

### PRIVATE SUPPORT FUNCTIONS ###

function _load_dot_env() {
  cd "${ROOT_DIR}"
  if [ -f ".env" ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source ".env"
    set +o allexport
  fi
}

function _py_build() {
  _load_dot_env
  directory=$1
  cp "${ROOT_DIR}/data.proto" "${ROOT_DIR}/cogment.yaml" "${ROOT_DIR}/${directory}"
  pushd "${ROOT_DIR}/${directory}"
  virtualenv -p python3 .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -r requirements.txt
  python -m cogment.generate
  deactivate
  popd
}

function _py_test() {
  _load_dot_env
  directory=$1
  pushd "${ROOT_DIR}/${directory}"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pytest
  deactivate
  popd
}

function _py_start() {
  _load_dot_env
  directory=$1
  pushd "${ROOT_DIR}/${directory}"
  .venv/bin/python -m main
  popd
}

function _run_parallel() {
  commands=("$@")
  commands_num="${#commands[@]}"

  # Using GNU parallel to launch the provided commands in parallel and properly handle
  # their output and termination.
  #
  # In particular:
  # - When this process ends, the commands will end (default behavior)
  # - When one of the commands fails everything is stopped (`--halt now,fail=1`)
  # - All the output are printed as they are generated by the command (`-u`)
  # - As many jobs as there are provided commands are ran (`-j "${commands_num}"`)
  parallel -j "${commands_num}" -u --halt now,fail=1 "${RUN_SCRIPT}" ::: "${commands[@]}"
}

function _run_sequence() {
  commands=("$@")

  for command in "${commands[@]}"; do
    "${command}"
  done
}

### GENERIC PUBLIC COMMANDS ###

function commands() {
  all_commands=$(declare -F | awk '{print $NF}' | sort | grep -Ev "^_")
  for command in "${all_commands[@]}"; do
    if [[ ! ("${command}" =~ ^_.*) ]]; then
      printf "%s\n" "${command}"
    fi
  done
}

### PROJECT SPECIFIC PUBLIC COMMANDS ###

function base_python_build() {
  cp "${ROOT_DIR}/data.proto" "${ROOT_DIR}/cogment.yaml" "${ROOT_DIR}/base_python"
  cp "${ROOT_DIR}/run_api.proto" "${ROOT_DIR}/base_python/cogment_verse/api/"
  pushd "${ROOT_DIR}/base_python"
  virtualenv -p python3 .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -e . # This is a reusable package, it needs to install itself
  pip install -r requirements.txt
  python -m cogment.generate
  python -m grpc.tools.protoc --proto_path=. --python_out=. --grpc_python_out=. ./cogment_verse/api/run_api.proto
  deactivate
  popd
}

function base_python_test() {
  _py_test base_python
}

function client_build() {
  _py_build client
  pushd "${ROOT_DIR}/client"
  cp "${ROOT_DIR}/run_api.proto" "./"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m grpc.tools.protoc --proto_path=. --python_out=. --grpc_python_out=. ./run_api.proto
  deactivate
  popd
}

function client() {
  _load_dot_env
  pushd "${ROOT_DIR}/client"
  COGMENT_VERSE_RUN_PARAMS_PATH="${ROOT_DIR}/run_params.yaml" .venv/bin/python -m main "$@"
}

function atari_roms_install() {
  _load_dot_env
  pushd "${ROOT_DIR}/environment"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  if [ -d ".atari_roms" ]; then
    printf "environment/.atari_roms already exists, skipping atari roms import \n"
  else
    mkdir .atari_roms
    # Download the roms.
    curl -s http://www.atarimania.com/roms/Roms.rar --output .atari_roms/roms.rar
    # And unrar them
    unrar x -y -r .atari_roms/roms.rar .atari_roms/
    # import everything that is supported
    ale-import-roms .atari_roms
  fi
  deactivate
  popd
}

function environment_build() {
  _py_build base_python
  _py_build environment
  atari_roms_install
}

function environment_start() {
  _py_start environment
}

function environment_test() {
  _py_test environment
}

function tf_agents_build() {
  _py_build base_python
  _py_build tf_agents
}

function tf_agents_start() {
  _py_start tf_agents
}

function torch_agents_build() {
  _py_build base_python
  _py_build torch_agents
}

function torch_agents_start() {
  _py_start torch_agents
}

function torch_agents_test() {
  _py_test torch_agents
}

function root_build() {
  _load_dot_env
  virtualenv -p python3 "${ROOT_DIR}/.venv"
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
  pip install -r "${ROOT_DIR}/requirements.txt"
  deactivate
}

function lint() {
  _load_dot_env
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
  black --check --diff .
  find . -name '*.py' -not -path '*/.venv/*' -print0 | xargs -0 pylint -j 4
  deactivate
}

function lint_fix() {
  _load_dot_env
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
  black .
  deactivate
}

function mlflow_build() {
  root_build
}

function mlflow_start() {
  _load_dot_env
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
  mlflow server \
    --host 0.0.0.0 \
    --port "${COGMENT_VERSE_MLFLOW_PORT}" \
    --backend-store-uri "sqlite:///${ROOT_DIR}/data/mlflow/mlflow.db" \
    --default-artifact-root "${ROOT_DIR}/data/mlflow/"
  deactivate
}

function web_client_build() {
  _load_dot_env
  export PORT="${COGMENT_VERSE_WEBCLIENT_PORT}"
  export REACT_APP_ORCHESTRATOR_HTTP_ENDPOINT="${COGMENT_VERSE_ORCHESTRATOR_HTTP_ENDPOINT}"
  cp "${ROOT_DIR}/data.proto" "${ROOT_DIR}/cogment.yaml" "${ROOT_DIR}/web_client"
  cd "${ROOT_DIR}/web_client"
  npm install --no-audit
  npm run build
}

function web_client_start() {
  _load_dot_env
  export PORT="${COGMENT_VERSE_WEBCLIENT_PORT}"
  cd "${ROOT_DIR}/web_client"
  npm run start
}

function web_client_start_dev() {
  _load_dot_env
  export PORT="${COGMENT_VERSE_WEBCLIENT_PORT}"
  export REACT_APP_ORCHESTRATOR_HTTP_ENDPOINT="${COGMENT_VERSE_ORCHESTRATOR_HTTP_ENDPOINT}"
  cd "${ROOT_DIR}/web_client"
  npm run dev
}

function cogment_install() {
  mkdir -p "${COGMENT_DIR}"
  pushd "${COGMENT_DIR}"
  curl --silent -L https://raw.githubusercontent.com/cogment/cogment/main/install.sh | bash /dev/stdin --skip-install --version "${COGMENT_VERSION}"
  popd
}

function orchestrator_start() {
  _load_dot_env
  "${COGMENT_DIR}/cogment" services orchestrator \
    --actor_port="${COGMENT_VERSE_ORCHESTRATOR_PORT}" \
    --lifecycle_port="${COGMENT_VERSE_ORCHESTRATOR_PORT}" \
    --actor_http_port="${COGMENT_VERSE_ORCHESTRATOR_HTTP_PORT}" \
    --pre_trial_hooks="${COGMENT_VERSE_PRETRIAL_HOOK_ENDPOINT}"
}

function trial_datastore_start() {
  _load_dot_env
  "${COGMENT_DIR}/cogment" services trial_datastore \
    --port="${COGMENT_VERSE_TRIAL_DATASTORE_PORT}" \
    --memory_storage_max_samples_size=1073741824
}

function model_registry_start() {
  _load_dot_env
  "${COGMENT_DIR}/cogment" services model_registry \
    --port="${COGMENT_VERSE_MODEL_REGISTRY_PORT}" \
    --archive_dir="${ROOT_DIR}/data/model-registry" \
    --sent_version_chunk_size=2097152 \
    --cache_max_items=100
}

function build() {
  _run_sequence cogment_install root_build base_python_build client_build environment_build tf_agents_build torch_agents_build web_client_build
}

function test() {
  _run_sequence base_python_test environment_test torch_agents_test
}

function services_start() {
  _run_parallel orchestrator_start trial_datastore_start model_registry_start environment_start tf_agents_start torch_agents_start mlflow_start
}

### MAIN SCRIPT ###

available_commands=$(commands)
command=$1
if [[ "${available_commands[*]}" = *"$1"* ]]; then
  shift
  ${command} "$@"
else
  printf "Unknown command [%s]\n" "${command}"
  printf "Available commands are:\n%s\n" "${available_commands[*]}"
  exit 1
fi
