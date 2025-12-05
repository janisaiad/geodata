#!/bin/bash
# Version simplifiée pour watch

if command -v nvidia-smi &> /dev/null; then
    watch -n 1 -d 'echo "=== GPU Memory ===" && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv && echo "" && echo "=== CPU/RAM ===" && free -h && echo "" && echo "=== Top Python Processes ===" && ps aux | grep python | grep -v grep | head -5 | awk "{print \$2, \$6/1024\"MB\", \$11, \$12, \$13}"'
else
    watch -n 1 -d 'echo "=== CPU/RAM ===" && free -h && echo "" && echo "=== Top Python Processes ===" && ps aux | grep python | grep -v grep | head -5 | awk "{print \$2, \$6/1024\"MB\", \$11, \$12, \$13}"'
fi

