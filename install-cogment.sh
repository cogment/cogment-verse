#!/usr/bin/env bash

### FUNCTIONS

install_dir="/usr/local"
skip_install=0
include_api=0

function usage() {
  printf "\
    Download and install Cogment\n\n \
    Usage:\n \
      %s [--version X.Y.Z[.PRE]] [--arch ARCH] [--os OS] [--skip-install]\n\n \
      Requires root access unless '--skip-install' is specified.\n\n \
    Options:\n \
      --version X.Y.Z[.PRE]:    Target version, default is latest.\n \
      --arch ARCH:              Target system architecture, default is this machine's.\n \
      --os OS:                  Target operating system, default is this machine's.\n \
      --include-api:            Also download & install the protobuf definitions for Cogment gRPC API.\n \
      --install-dir:            Install dir (default is %s).\n \
      --skip-install:           Do not install the downloaded binary to the recommended location.\n \
      -h, --help:               Show this screen.\n" "$(basename "${BASH_SOURCE[0]}")" "${install_dir}"
}

VERSION_SED_REGEX="[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\(-[a-zA-Z0-9][a-zA-Z0-9]*\)\{0,1\}"

function validate_version() {
  local input_version=$1
  shift
  local parsed_version
  parsed_version=$(sed -n "s/^v\{0,1\}\(${VERSION_SED_REGEX}\)$/\1/p" <<<"${input_version}")
  printf %s "${parsed_version}"
}

function get_latest_gh_release() {
  local gh_repo=$1
  curl --silent "https://api.github.com/repos/${gh_repo}/releases/latest" | # Get latest release from GitHub api
    grep '"tag_name":' |                                                    # Get tag line
    sed -E 's/.*"([^"]+)".*/\1/'                                            # Pluck JSON value
}

### SCRIPT PROPER

set -o errexit

while [[ "$1" != "" ]]; do
  case $1 in
    --version)
      shift
      version=$1
      ;;
    --arch)
      shift
      arch=$1
      ;;
    --os)
      shift
      os=$1
      ;;
    --include-api)
      include_api=1
      ;;
    --install-dir)
      shift
      install_dir=$1
      ;;
    --skip-install)
      skip_install=1
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    *)
      printf "%s: unrecognized argument.\n" "$1"
      usage
      exit 1
      ;;
  esac
  shift
done

## 1 - Deal with the system architecture

if [[ -z "${arch}" ]]; then
  arch=$(uname -m)
fi

case ${arch} in
  "x86_64" | "amd64")
    arch="amd64"
    ;;
  *)
    printf "%s: unsupported system architecture.\n" "${arch}"
    exit 1
    ;;
esac

## 2 - Deal with the os

if [[ -z "${os}" ]]; then
  os=$(uname)
fi

case ${os} in
  "Linux" | "linux")
    os="linux"
    ;;
  "WindowsNT" | "windows")
    os="windows"
    ;;
  "Darwin" | "macos")
    os="macos"
    ;;
  *)
    printf "%s: unsupported operating system.\n" "${os}"
    exit 1
    ;;
esac

## 3 - Deal with the version
if [[ -z "${version}" ]]; then
  version=$(get_latest_gh_release "cogment/cogment")
else
  input_version=${version}
  version="v$(validate_version "${input_version}")"
  if [[ -z "${version}" ]]; then
    printf "%s: provided version is invalid.\n" "${input_version}"
    usage
    exit 1
  fi
fi

if [[ "${skip_install}" == 0 && $(/usr/bin/id -u) != 0 ]]; then
  printf "To install Cogment this script should run as a root user.\n"
  exit 1
fi

## 4 - Deal with cogment
cogment_url="https://github.com/cogment/cogment/releases/download/${version}/cogment-${os}-${arch}"
if [[ "${os}" == "windows" ]]; then
  cogment_url="${cogment_url}.exe"
  cogment_local_path="./cogment.exe"
  printf "Downloading Cogment from '%s'...\n" "${cogment_url}"
  curl -L --silent "${cogment_url}" --output "${cogment_local_path}"
  printf "Copy this file to a directory belonging to your PATH environment variable.\n"
else
  cogment_local_path="./cogment"
  printf "Downloading Cogment from '%s'...\n" "${cogment_url}"
  curl -L --silent "${cogment_url}" --output "${cogment_local_path}"
  if [[ "${skip_install}" == 1 ]]; then
    chmod +x "${cogment_local_path}"
    printf "Cogment downloaded, test it by running '%s version'.\n" "${cogment_local_path}"
  else
    cogment_installed_path="${install_dir}/bin/cogment"
    mv "${cogment_local_path}" "${cogment_installed_path}"
    chmod +x "${cogment_installed_path}"
    printf "Cogment installed at '%s', test it by running 'cogment version'.\n" "${cogment_installed_path}"
  fi
fi

## 5 - Deal with cogment api
cogment_api_url="https://github.com/cogment/cogment/releases/download/${version}/cogment-api.tar.gz"
if [[ "${include_api}" == 1 ]]; then
  cogment_api_local_path="./cogment-api"
  printf "Downloading Cogment API from '%s'...\n" "${cogment_api_url}"
  mkdir -p "${cogment_api_local_path}"
  curl -L --silent "${cogment_api_url}" | tar xf - --directory="${cogment_api_local_path}"
  if [[ "${skip_install}" == 1 ]]; then
    printf "Cogment API downloaded to '%s'" "${cogment_api_local_path}"
  else
    cogment_installed_path="${install_dir}/include"
    mv "${cogment_local_path}/*" "${cogment_installed_path}"
    rm -rf "${cogment_api_local_path}"
    printf "Cogment installed at '%s', test it by running 'cogment version'.\n" "${cogment_installed_path}"
  fi
fi
