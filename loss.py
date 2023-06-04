import numpy as np
import torch
from config import CFG
import torch.nn.functional as F 
device_loss = CFG.device
Logged_oneDiSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalization factor for Gaussians


def pairwise_distance(preds,gts):

    preds_n = preds[:,:,0:2].contiguous().view(-1, 2).to(device_loss)   # change to 2 dims
    gts_n = gts[:,:,0:2].contiguous().view(-1, 2).to(device_loss)     # change to 2 dims

    return torch.mean(F.pairwise_distance(preds_n, gts_n))

def Mean_squared_distance(preds,gts):
    error = (preds - gts)**2
    # print("ERRRRRRRRRRR:", torch.mean(error))
    return torch.mean(error)
    # 
def Bivariate(pi,sigma_x,sigma_y, mu_x , mu_y,input,device = device_loss):

    # Check the num of dims
    if input.ndim ==3:
        x = input[:,:,0].to(device)
        y = input[:,:,1].to(device)
        x = x.unsqueeze(-1).to(device)
        y = y.unsqueeze(-1).to(device)
        #print("Num of Dims is 3 : ",input.shape)
    elif input.ndim ==2:
        x = input[:,0]
        y = input[:,1]
        x = x.unsqueeze(dim=1).to(device)
        y = y.unsqueeze(dim=1).to(device)
        # print("Num of Dims is 2 : ",input.shape)
    # make |mu|=K copies of y, subtract mu, divide by sigma
    #print("Input: ",input.shape ,"\nX: ",x.shape,"\nY: ",y.shape,"\nMu_x : ",mu_x.shape,"\nMu_y : ",mu_y.shape,"\nSigma_x : ",sigma_x.shape)
    result_x = torch.square((x.expand_as(mu_x) - mu_x) * torch.reciprocal(sigma_x))
    result_y = torch.square((y.expand_as(mu_y) - mu_y) * torch.reciprocal(sigma_y))
    

    result = -0.5*(result_x + result_y)
    log_pi = torch.log(pi)
    log_TwoPiSigma = -torch.log (2.0*np.pi*sigma_x*sigma_y)
    # expand log values
    values = log_pi + log_TwoPiSigma.expand_as(log_pi) 

    return (values + result)


# Define the custom loss function
counter = 0
def mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,y,mixtures,device=device_loss):
    # calculate the score for each mixture of the gaussian_distribution
    # input shape (sample_size,num_mixtures,parameter) parametr is 2 in mue (x,y) and 2,2 in sigma [xx,xy,yx,yy] 
    # Pi has shape of  (sample_size,num_mixtures)
    # swap axis to have shape (num_mixtures,sample_size,parameter)
    # print("Before anythinG: ",sigma_x.shape,sigma_y.shape, mu_x.shape , mu_y.shape,'\n: ',y.shape,'\n')


    # mask = torch.lt(pi, 0)
    # mask_res = torch.lt(pi, 0)

    # # check if any element in the tensor satisfies the condition
    # if torch.any(mask):
    #     print("The pi tensor contains negative values.")
    # else:
    #     print("The pi tensor does not contain negative values.")

    
    result = Bivariate(pi,sigma_x,sigma_y, mu_x , mu_y,y,device) 
    # print("result shape: ",result.shape)
    # mask_res = torch.lt(result, 0)

    # # check if any element in the tensor satisfies the condition
    # if torch.any(mask_res):
    #     print("The result tensor contains negative values.")
    # else:
    #     print("The result tensor does not contain negative values.")
    # max of results
    # m = torch.max(result)
    # changed value of max
    #torch.tensor
    m = (torch.max(result, dim=2, keepdim=True)[0]).repeat(1,1,mixtures)
    # print("max of results shape: ",m.shape)
    # print("result of results shape: ",result.shape)
    # LogSumExp trick log(sum(exp)) will be = m + log(sum (exp (result-m)))
    exp_value = torch.exp(result-m)
    # print("exp_value of exp_value shape: ",exp_value.shape)
    epsilon = 0.00001
    # changed the last dimention dim from 1 to -1
    result = torch.sum(exp_value, dim=-1) + epsilon
    #print("result after sum: ",result)
    #org
    #result = -(m + torch.log(result))
    result = -(m[:,:,0] + torch.log(result))
    global counter 
    counter+=1
    if(torch.isnan(result).any()):
        print("Counter loss: ",counter)
        print("result m : ",m.item)
    return torch.mean(result)

