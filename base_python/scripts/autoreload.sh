#!/usr/bin/env bash
# Start an autoreload dev server using watchmedo (https://github.com/gorakhargosh/watchdog)

set -o errexit

# touch `data.proto` in 1 second to trigger the autoreload
(sleep 1.0 && touch data.proto)&

# Copy the 'autoreload.yaml' file to the working directory
SCRIPT_DIRECTORY=$(dirname "${BASH_SOURCE[0]}")
cp ${SCRIPT_DIRECTORY}/autoreload.yaml ./autoreload.yaml 
watchmedo tricks autoreload.yaml
