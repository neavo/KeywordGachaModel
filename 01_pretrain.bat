@echo off
@chcp 65001 > nul

set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

cd /d %~dp0
call python 01.py

pause