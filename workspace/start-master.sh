#!/bin/bash
python /opt/workspace/main.py --epochs=20 --batch-size=1024 --backend=nccl --save-model
