@echo off
REM Launch the MIP tool from the "squidxplorer" conda environment.
REM Double-click this file, or pass an acquisition folder:  mip-tool.bat "C:\path\to\acquisition"

call conda activate squidxplorer 2>nul
if errorlevel 1 (
    echo Could not activate the 'squidxplorer' conda environment.
    echo First-time setup, from the SquidXplorer folder:
    echo     conda env create -f environment.yml
    pause
    exit /b 1
)

squidxplorer-view %*
