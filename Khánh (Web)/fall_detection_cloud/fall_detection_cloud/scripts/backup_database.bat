@echo off
REM PostgreSQL Database Backup Script for Fall Detection System
REM Run this script daily via Windows Task Scheduler

SET BACKUP_DIR=C:\backups\postgres
SET CONTAINER_NAME=fall_db
SET DB_NAME=falldetection
SET DB_USER=postgres
SET RETENTION_DAYS=7

REM Create backup directory if not exists
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

REM Generate timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
SET TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

REM Backup database
echo Backing up database %DB_NAME%...
docker exec %CONTAINER_NAME% pg_dump -U %DB_USER% %DB_NAME% > "%BACKUP_DIR%\falldetection_%TIMESTAMP%.sql"

if %ERRORLEVEL% EQU 0 (
    echo Backup successful: falldetection_%TIMESTAMP%.sql
) else (
    echo Backup failed!
    exit /b 1
)

REM Clean up old backups (keep last RETENTION_DAYS days)
echo Cleaning up backups older than %RETENTION_DAYS% days...
forfiles /P "%BACKUP_DIR%" /M falldetection_*.sql /D -%RETENTION_DAYS% /C "cmd /c del @path"

echo Backup completed.
