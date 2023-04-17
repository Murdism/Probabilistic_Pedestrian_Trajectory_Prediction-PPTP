# Probablistic Trajectory Prediction Model
The main task of this code is to predict the trajectory of a pedestrian.
We predict multimodal trajectory with probabilites.

# Usage

## Data setup
### Dataset
The dataset folder must have the following structure:

    - dataset
      - dataset_name
        - train_folder
        - test_folder
        - validation_folder 
We use part of the dataloader in Trajectory-Transformer,"baselineUtils.py"
''' git clone  https://github.com/FGiuliari/Trajectory-Transformer '''
### Pre-Trained 
Model_x_y_z --> Here, x represents number of epochs while x and y are the ADE and FDE related.

- MODEL without roman numbering is combination of GMM and MSQ loss
- MODEL II is mdn loss only 
- MODEL III is mdn and msq loss

[Download Pre-Trained] (https://drive.google.com/file/d/13So1tsDC6gm8ULDtRoBD8UJoHgbRMSir/view?usp=sharing)
## Main file
To train and test just run the *GMM_Transformer.ipynb* with different configuration from *config.py*

### config.py
Most parameters including your choice to test or train can be set up on this file:

It has two classes:

- *class CFG*: contains main parameters that are used to train the model.
Including the number of heads of transformer,number of batchs, path of model (to be saved)

- *class Args*: contains mainly arguments related to dataset choice and mode (train/test), visuzlize output or not and so on.




# SOURCES
[MDN_Thesis] (https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-DNN-Regression.ipynb)

[Theano] (https://tensorcruncher.wordpress.com/2016/09/07/mdnmixture-density-network-implementation-in-theano/)

[Train] https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

[ELU] (https://deeplearninguniversity.com/elu-as-an-activation-function-in-neural-networks/) activation and Laplacian Distribution