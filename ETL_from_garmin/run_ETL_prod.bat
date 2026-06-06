@echo off
cd /d "D:\Personal\Ingeniero\03. Data science\01. Running\garmin-running-analytics\"
set PYTHONPATH=%cd%
call .venv\Scripts\activate
python ETL_from_garmin\main_ETL_from_garmin.py prod
pause