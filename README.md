# Probablistic Trajectory Prediction Model
The main task of this code is to predict the trajectory of a pedestrian.
We predict multimodal trajectory with probabilites.

# Usage
## Requirements 
In this project we use *python 3.10.9* and *pytorch 1.12.1 with cuda 10.2.1 (cudnn 7.6.5)*.

**Note**: If there is no GPU or no CUDA, change the device argument to cpu in the config file.  

Other requirements can be found in the requirements file.  

First you need to create an environment (Conda is used here):

```conda create -n env_name python==3.10.9```  

Then activate the environment:

```conda activate env_name```  

Finally, to install the requirements use:

```pip install requirements.txt ```

## Data setup
### Dataset
The dataset folder must have the following structure:

    - dataset
      - dataset_name
        - train_folder
        - test_folder
        - validation_folder 
We use part of the dataloader in Trajectory-Transformer,"baselineUtils.py"

The training data input has shape *Num_batchs,Batch_size,obs_traj_length,num_features*.
The obs_traj_length is set to 8 by default and the number of features can be changed based on preference in the config file.
The number of features changes due to preprocessing, the original input has two features x and y.   

The output has shape *Num_batchs,Batch_size,obs_traj_length,num_features*. Here num_features is 2; like the input.

``` git clone  https://github.com/FGiuliari/Trajectory-Transformer ```
### Pre-Trained 
Model_x_y_z --> Here, x represents number of epochs while x and y are the ADE and FDE related.

- MODEL without roman numbering is combination of GMM and MSQ loss
- MODEL II is mdn loss only 
- MODEL III is mdn and msq loss

[Download Pre-Trained](https://drive.google.com/file/d/13So1tsDC6gm8ULDtRoBD8UJoHgbRMSir/view?usp=sharing)
## Configuration 
To train and test just run the *GMM_Transformer.ipynb* with different configuration from *config.py*

### config.py
Most parameters including your choice to test or train can be set up on this file:

It has two classes:

- *class CFG*: contains main parameters that are used to train the model.
Including the number of heads of transformer,number of batchs, path of model (to be saved)

- *class Args*: contains mainly arguments related to dataset choice and mode (train/test), visuzlize output or not and so on.


## Training
- *Train.py* : contains functions used for training. 
- To Train *class CFG* and *class Args* can be edited. 
  - *class CFG* : num_heads,num_encoders,num_epochs_num_batchs,num_features ...etc.
  - *class Args* : change mode to 'train', 
## Testing
- *Test.py* : contains functions used for testing.
- To test *class CFG* and *class Args* can be edited. 
  - *class CFG* : num_heads,num_encoders,num_epochs_num_batchs,num_features ...etc.
  - *class Args* : change mode to 'test', use model_path to use pre-trained model 



# SOURCES
[MDN_Thesis](https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-DNN-Regression.ipynb)

[Theano](https://tensorcruncher.wordpress.com/2016/09/07/mdnmixture-density-network-implementation-in-theano/)

[Train](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec)

[ELU](https://deeplearninguniversity.com/elu-as-an-activation-function-in-neural-networks/) 