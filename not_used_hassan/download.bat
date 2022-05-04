@echo off
echo This will take a while, if you are sure press any key to continue, otherwise close this window
pause()
python .\data_retrieval\video_downloader.py
echo Download Complete
echo Now run preprocess.py with the venv active
pause()
