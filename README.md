# Probablistic Trajectory Prediction Model
The main task of this code is to predict the trajectory of a pedestrian.
We predict multimodal trajectory with probabilites.

# Usage

## Data setup
The dataset folder must have the following structure:

    - dataset
      - dataset_name
        - train_folder
        - test_folder
        - validation_folder 
We use part of the dataloader in Trajectory-Transformer,"baselineUtils.py"
''' git clone  https://github.com/FGiuliari/Trajectory-Transformer '''

## Main file
To train and test just run the *GMM_Transformer.ipynb* with different configuration from *config.py*

### config.py
Most parameters including your choice to test or train can be set up on this file:
It has two classes
*class CFG*: contains main parameters that are used to train the model.
Including the number of heads of transformer,number of batchs, path of model (to be saved)
*class Args*: contains mainly arguments related to dataset choice and mode (train/test), visuzlize output or not and so on.

[Download Pre-Trained] (https://drive.google.com/file/d/13So1tsDC6gm8ULDtRoBD8UJoHgbRMSir/view?usp=sharing)


## SOURCES
[MDN_Thesis] (https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-DNN-Regression.ipynb)

[Theano] (https://tensorcruncher.wordpress.com/2016/09/07/mdnmixture-density-network-implementation-in-theano/)

[Train] https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

[ELU] (https://deeplearninguniversity.com/elu-as-an-activation-function-in-neural-networks/) activation and Laplacian Distribution



MODEL II is mdn loss only 
MODEL III is mdn and msq loss








Taking the negative logarithm of the mixture density function of a bivariate normal distribution (-log f(x, y)) is a common operation in machine learning and statistical modeling, particularly when using maximum likelihood estimation to fit the mixture model to data.

The negative logarithm of the mixture density function is equal to the negative logarithm of the product of the individual bivariate normal distributions in the mixture, which can be simplified to a sum of negative logarithms of the individual densities:

-log f(x, y) = -log(∑ᵢ wᵢ * ϕ(x, y | μᵢ, Σᵢ))
= -log(∑ᵢ exp(-log wᵢ) * ϕ(x, y | μᵢ, Σᵢ))
= -log(exp(-m) * ∑ᵢ exp(-log wᵢ -m) * ϕ(x, y | μᵢ, Σᵢ))
= m - log(∑ᵢ exp(-log wᵢ -m) * ϕ(x, y | μᵢ, Σᵢ))

where:

m = max(-log(wᵢ * ϕ(x, y | μᵢ, Σᵢ))) is a constant used to improve numerical stability by preventing the exponentiation of very large or very small values.
Taking the negative logarithm of the mixture density function has some benefits over using the original mixture density function. First, it allows us to simplify the computation by converting the product of densities to a sum of logarithms of densities, which is more numerically stable. Second, it transforms the problem of maximizing the likelihood of the mixture model to minimizing the negative logarithm of the likelihood, which is a convex optimization problem and can be solved using standard optimization algorithms.

The negative logarithm of the mixture density function should always be non-negative, since the mixture density function is non-negative by definition. If there are negative values in the negative logarithm of the mixture density function, it could be due to numerical precision errors or other issues with the implementation.

