@echo off
@chcp 65001 > nul

@REM 设置工作目录
cd /d %~dp0

@REM 检查是否有参数传递给脚本
if "%~1"=="" (
    echo 请拖放一个文件到此批处理文件上 ...
    goto :END
)

@REM 执行python脚本
call python 99.py %1

:END
pause