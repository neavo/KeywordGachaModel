@echo off
@chcp 65001 > nul

cd /d %~dp0
call python 04.py

pause