@echo off
call .venv\Scripts\activate
python ETL_from_garmin\main_ETL_from_garmin.py prod
pause