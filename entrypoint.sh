#!/bin/bash

if [[ "$1" == "--dac" ]]; then
    echo "Running DAC setup..."
    cd /depth_any_camera/dac/models/ops && pip install -e . && cd /source/splat_one && bash
else
    echo "Default command executed"
    exec "$@"
fi
