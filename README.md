# NISTADS: AI-driven Adsorption Prediction

## Project Overview
This project delves into the fascinating world of adsorption, a surface-based process where a film of particles (adsorbate) accumulate on the surface of a material (adsorbent). This phenomenon plays a pivotal role in numerous industries. For instance, it is widely applied in water treatment facilities for the purification of water, in air filters to improve air quality, and in the automotive industry within catalytic converters to reduce harmful emissions. The adsorption of compounds is usually quantified by measuring the adsorption isotherm a given adsorbate/adsorbent combination. The objective of this project is to harness the power of machine learning to predict the adsorption of chemicals on adsorbent materials. The aim is to build a model that can accurately predict the adsorbed amount of a specific guest-host combination under various conditions, by leveraging the data from the NIST/ARPA-E Database. This could have significant implications for industries that rely on these materials, potentially leading to more efficient processes and better materials design. As such, this project takes a different approach compared to fitting adsorption data with theoretical model for adsorption constants calculation, instead proposing the use of a deep learning approach to understand adsorption isotherm patterns by leveraging a large volume of experimental data.

## Project data

### NIST/ARPA-E Database
The NIST/ARPA-E Database of Novel and Emerging Adsorbent Materials is a free, web-based catalog of adsorbent materials and measured adsorption properties of numerous materials obtained from article entries from the scientific literature. Search fields for the database include adsorbent material, adsorbate gas, experimental conditions (pressure, temperature), and bibliographic information (author, title, journal), and results from queries are provided as a list of articles matching the search parameters. The database also contains adsorption isotherms digitized from the cataloged articles, which can be compared visually online in the web application or exported for offline analytics. The data used for the machine learning training has been extracted using another python application, which is available at the following repository: https://github.com/CTCycle/NISTADS-data-collection. This script makes use of the NIST/ARPA-E Database API to collect adsorption isotherm data for both single component experiments and binary mixture experiments, and conveniently splitting those in two different datasets (Single Component ADSorption and Binary Mixture ADSorption). Since NISTADS is focused on predicting single component adsorption isotherms, it will make use of the single component dataset (SCADS) for the model training.

### Data preprocessing
The data undergoes preprocessing via a tailored pipeline, which initially filters out undesired experiments. These include experiments featuring negative values for temperature, pressure, and uptake, or those falling outside predefined boundaries for pressure and uptake ranges (refer to configurations for specifics). Pressure and uptakes are standardized to a uniform unit—Pascal for pressure and mol/g for uptakes. Following this refinement, the physicochemical attributes of the absorbate species are unearthed through the PUBCHEM API. This enriches the input data with molecular properties such as molecular weight, the count of heavy atoms, covalent units, and H-donor/acceptor statistics. Subsequently, features pertaining to experiment conditions (e.g., temperature) and adsorbate species (physicochemical properties) are normalized. Names of adsorbents and adsorbates are encoded into integer indices for subsequent vectorization by the designated embedding head of the model. Pressure and uptake series are also normalized, utilizing upper boundaries as the normalization ceiling. Additionally, initial zero measurements in pressure and uptakes series are removed to mitigate potential bias towards zero values. Finally, all sequences are reshaped to have the same length using post-padding with a specified padding value (defaulting to -1 to avoid conflicts with actual values) and then normalized.

### SCADS model
...

## Installation 
First, ensure that you have Python 3.10.12 installed on your system. Then, you can easily install the required Python packages using the provided requirements.txt file:

`pip install -r requirements.txt` 

In addition to the Python packages, certain extra dependencies may be required for specific functionalities. These dependencies can be installed using conda or other external installation methods, depending on your operating system. Specifically, you will need to install graphviz and pydot to enable the visualization of the 2D model architecture:
- graphviz version 2.38.0
- pydot version 1.4.2

You can install these dependencies using the appropriate package manager for your system. For instance, you might use conda or an external installation method based on your operating system's requirements.

### CUDA GPU Support (Optional, for GPU Acceleration)
If you have an NVIDIA GPU and want to harness the power of GPU acceleration using CUDA, please follow these additional steps. The application is built using TensorFlow 2.10.0 to ensure native Windows GPU support, so remember to install the appropriate versions:

#### 1. Install NVIDIA CUDA Toolkit (Version 11.2)
To enable GPU acceleration, you'll need to install the NVIDIA CUDA Toolkit. Visit the [NVIDIA CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads) and select the version that matches your GPU and operating system. Follow the installation instructions provided. Alternatively, you can install `cuda-toolkit` as a package within your environment.

#### 2. Install cuDNN (NVIDIA Deep Neural Network Library, Version 8.1.0.77)
Next, you'll need to install cuDNN, which is the NVIDIA Deep Neural Network Library. Visit the [cuDNN download page](https://developer.nvidia.com/cudnn) and download the cuDNN library version that corresponds to your CUDA version (in this case, version 8.1.0.77). Follow the installation instructions provided.

#### 3. Additional Package (If CUDA Toolkit Is Installed)
If you've installed the NVIDIA CUDA Toolkit within your environment, you may also need to install an additional package called `cuda-nvcc` (Version 12.3.107). This package provides the CUDA compiler and tools necessary for building CUDA-enabled applications.

By following these steps, you can ensure that your environment is configured to take full advantage of GPU acceleration for enhanced performance.

#### 4. Additional Package for XLA Acceleration
XLA is designed to optimize computations for speed and efficiency, particularly beneficial when working with TensorFlow and other machine learning frameworks that support XLA. By incorporating XLA acceleration, you can achieve significant performance improvements in numerical computations, especially for large-scale machine learning models. XLA integration is directly available in TensorFlow but may require enabling specific settings or flags. To enable XLA acceleration globally across your system, you need to set an environment variable named `XLA_FLAGS`. The value of this variable should be `--xla_gpu_cuda_data_dir=path\to\XLA`, where `path\to\XLA` must be replaced with the actual directory path that leads to the folder containing the nvvm subdirectory. It is crucial that this path directs to the location where the file `libdevice.10.bc` resides, as this file is essential for the optimal functioning of XLA. This setup ensures that XLA can efficiently interface with the necessary CUDA components for GPU acceleration.

## How to use
The user can navigate the project folder to find different subfolders, depending on the desired task to perform. The `component\` directory serves as the home for fundamental files crucial for the flawless operation of the program. It's imperative not to tamper with these files as any modifications could prevent the script from properly working.

**Data**
This folder contains the data utilized for the model training. The main files present in this location are:
- `adsorbates_dataset.csv` provides information about the adsorbates species
- `adsorbents_dataset.csv` provides information about the adsorbent materials
- `SCADS_dataset.csv` contains the data that will be used for SCADS training
Execute `data_validation.py` to conduct an in-depth analysis leveraging the adsorption isotherm dataset.

**Model**
In this folder are the necessary files for conducting model training and evaluation:
- `model/checkpoints` acts as the default repository where checkpoints of pre-trained models are stored.
- Run `model_training.py` to initiate the training process for deep learning models.
- Run `model_evaluation.py` to evaluate the performance metrics of pre-trained models.

**Inference**
Use `adsorption_predictions.py` from this directory to predict adsorption given empirical pressure series and parameters. The predicted values are saved in `SCADS_predictions.py`. 

### Configurations
The configurations.py file allows to change the script configuration. The following parameters are available:

**Advanced settings for training:**
- `use_mixed_precision:` whether or not to use mixed precision for faster training (mix float16/float32)
- `use_tensorboard:` activate or deactivate tensorboard logging
- `XLA_acceleration:` use of linear algebra acceleration for faster training 
- `training_device:` select the training device (CPU or GPU)
- `num_processors:` number of processors (cores) to be used during training; if set to 1, multiprocessing is not used

**Settings for training routine:**
- `epochs:` number of training iterations
- `learning_rate:` learning rate of the model 
- `batch_size:` size of batches to be fed to the model during training

**Model settings:**
- `embedding_dims:` embedding dimensions for guest and host inputs
- `generate_model_graph:` generate and save 2D model graph (as .png file)

**Settings for training data:**
- `num_samples:` number of experiments to consider for training
- `test_size:` size of the test dataset (fraction of total samples)
- `pad_length:` max length of the pressure/uptake series (for padding)
- `pad_value:` number to assign to padded values (default: -1)
- `min_points:` minimum amount of measurements points for an experiment to be selected
- `max_pressure:` maximum allowed pressure (in Pascal)
- `max_uptake:` maximum allowed uptake (in mol/g)

**General settings:**
- `seed:` global random seed

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.
