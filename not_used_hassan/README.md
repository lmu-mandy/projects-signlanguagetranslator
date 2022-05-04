# Word Level ASL Classifier

## Requirements

* PyTorch (with CUDA support, if possible)
* PyTorch-Lightning
* PyTorchVideo
* TorchMetrics
* Scikit-Learn
* Numpy (included in PyTorch)
* OpenCV-Python (a.k.a cv2) (with CUDA support, if possible)

## Venv Setup

1. Install Python 3.9 for your Operating System
    * Go to the link below and download/install Python 3.9.10 for your operating system, and make sure to check the “Add Python to PATH” option, also take note of Python’s installation directory, and have it accessible for later.
    * <https://www.python.org/downloads/release/python-3910/>
2. Create Virtual Python Environment
    1. Open a command terminal in the project directory using the cd command
    2. To install the venv module type into the command terminal: `python -m pip install virtualenv`
    3. To create a virtual python environment in your terminal’s current working directory type: `python -m venv ./NAME`  
    Where `NAME` is the name of your virtual environment (usually people name it venv, which from this point out I will refer to as venv)
    4. To activate, or start using, your new venv, type in the following command in the terminal:  
       * `& ./venv/Scripts/activate` (if using Windows Command Prompt)
       * `& ./venv/Scripts/Activat.ps1` (if using Microsoft PowerShell)
    5. To exit, or deactivate, the venv, simply enter `deactivate` into your terminal
3. Install PyTorch
    1. Go to <https://pytorch.org/get-started/locally/> and select “Stable”, then your operating system, (I used Windows), then “Pip”, “Python”, then if you are using an NVidia GPU, select “CUDA 11.3”, otherwise select “CPU”.
    2. Copy the Command generated at the bottom of the table
        An example command for Windows and CUDA 11.3 would be:
        `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113`
4. Installing OpenCV-Python with CUDA for Windows
    1. Follow this guide, starting from “Step3 | Download & install Visual Studio”, then come back here & do step 4.2  
        <https://thinkinfi.com/install-opencv-gpu-with-cuda-for-windows-10/>
    2. Moving CV2 folder to your Virtual Python Environment
        1. Go to your Python installation directory that you took note of in Step 1.a.
        2. Then navigate to the ‘...\Python39\Lib\site-packages\’ directory
        3. Copy the ‘cv2’ folder from the above directory into your virtual environment’s ‘site-packages’ directory, which can be found in ‘...\venv\Lib\site-packages\’
5. Installing Remaining Modules
    1. While in the Virtual Python Environment enter the following command, and if asked to choose between Yes or No, always select Yes  
    `pip install -U torchmetrics pytorch-lightning pytorchvideo scikit-learn numpy cyhunspell keras pandas matplotlib tensorflow catboost lightgbm seaborn pyforest`

## Data Retrieval

With the terminal open in the `not_used_hassan` folder and the __venv active__, enter the following command:  
`python ./data_retrieval/video_downloader.py`  
To preprocess the data:  
`python ./preprocess.py`  
To run the model:  
`python ./model2.py`  
The model currently doesn’t train correctly.

## Reference Links & Credits

* WLASL Dataset: <https://dxli94.github.io/WLASL/>
* `VideoResNet.py` was retrieved in the PyTorch API Github
* The scripts in the `data_retrieval` folder were originally from the WLASL Dataset github, but were slightly modified for this project
* The rest of the Python scripts (`imports.py model1.py model2.py model3.py preprocess.py`) were done by Hassan Hage Hassan
