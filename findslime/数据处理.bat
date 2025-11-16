@echo off
REM 设置虚拟环境目录
set VENV_DIR=venv

REM 检查虚拟环境是否存在
if not exist %VENV_DIR% (
    echo 创建虚拟环境...
    python -m venv %VENV_DIR%
)

call %VENV_DIR%\Scripts\activate


REM 运行你的 Python 脚本
echo 正在运行脚本...
python data.py

REM 退出虚拟环境
venv\Scripts\deactivate
echo 任务完成！
pause