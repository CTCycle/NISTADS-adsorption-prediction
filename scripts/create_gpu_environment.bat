@echo off
rem Use this script to create a new environment called "SCADS"

echo STEP 1: Creation of SCADS environment
call conda create -n SCADS python=3.10 -y
if errorlevel 1 (
    echo Failed to create the environment SCADS
    goto :eof
)

rem If present, activate the environment
call conda activate SCADS

rem Install additional packages with pip
echo STEP 2: Install python libraries and packages
call pip install numpy pandas scikit-learn matplotlib tensorflow==2.10 
if errorlevel 1 (
    echo Failed to install Python libraries.
    goto :eof
)

rem Install CUDA and cuDNN via conda from specific channels
echo STEP 3: Install conda libraries for CUDA GPU support
call conda install conda-forge::cudatoolkit nvidia/label/cuda-12.0.0::cuda-nvcc conda-forge::cudnn -y
if errorlevel 1 (
    echo Failed to install CUDA toolkits.
    goto :eof
)

rem Install additional tools
echo STEP 4: Install additional libraries
call conda install graphviz -y
call pip install pydot
if errorlevel 1 (
    echo Failed to install Graphviz or Pydot.
    goto :eof
)

rem Print the list of dependencies installed in the environment
echo List of installed dependencies
call conda list

set/p<nul =Press any key to exit... & pause>nul
