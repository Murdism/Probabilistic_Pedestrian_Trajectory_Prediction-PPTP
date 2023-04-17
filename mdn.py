import torch 
import numpy as np
from config import CFG

device = CFG.device
batch_size = CFG.batch_size
def sample_mean(pi, sigma, mue, mode = "MaxMin"):

    # if mode == "MaxMin":
    # print("sigma shape",sigma.shape)
    # print("sigma shape",sigma.shape)    
    max_indices = torch.argmax(pi, dim=2).unsqueeze(-1)
    # use gather to select the mix dimension of a based on the indices in b_squeezed
    selected_mix = torch.gather(mue, dim=2, index=max_indices.unsqueeze(dim=-1).repeat(1, 1, 1, 2))

    # squeeze out the mix dimension to get the result of shape (batch_size, seq_len, 2)
    selected_mix = selected_mix.squeeze(dim=2)
    
    return selected_mix
def sampler_all_mean(pi, sigma_x,sigma_y, mu_x , mu_y,gt,mixtures=5,n_samples = 10,use_mean = 'all',postprocess=True):

    # use_mean = 'best'  ---uses onlty one mean
    # use_mean = 'all'   ---- uses all means
    pred_all,gt_all,pi_all = [],[],[]
    seq_pred , seq_gt = [] , []
    batch_pred , batch_gt = [] , []
    for pi_seq, sigma_x_seq,sigma_y_seq, mu_x_seq, mu_y_seq,gt_seq in zip(pi, sigma_x,sigma_y, mu_x , mu_y,gt):
        point_pred = []
        point_gt = []
        for  pi_point, sigma_x_point,sigma_y_point,mu_x_point, mu_y_point,gt_point in zip ( pi_seq, sigma_x_seq,sigma_y_seq, mu_x_seq, mu_y_seq,gt_seq):
            
                pi_value = pi_point.detach().cpu().numpy()   
                if use_mean=='best':
                    k = np.argmax(pi_value)
                    pred =[mu_x_point[k].detach().cpu().numpy(),mu_y_point[k].detach().cpu().numpy()]
                    point_pred.append(np.array(pred))
                    point_gt.append(np.array(gt_point))
                
                elif use_mean=='all':
                    mixture_mean = []
                    gt_mean = []
                    for k,weight in enumerate(pi_value):
                        print("AT: ",k,"weight: ",weight,"mu : ",mu_x_point[k].detach().cpu().numpy(),mu_y_point[k].detach().cpu().numpy())
                        if (weight < -1):
                            pred =np.array([0,0])   
                        else:
                            pred =[mu_x_point[k].detach().cpu().numpy(),mu_y_point[k].detach().cpu().numpy()]
                        
                        mixture_mean.append(np.array(pred))
                        gt_mean.append(np.array(gt_point))
                    print("Added: " ,mixture_mean)
                    point_pred.append(np.array(mixture_mean))
                    point_gt.append(np.array(gt_mean))

                elif (n_samples >1):
                    mixture_mean = []
                    gt_mean = []  
                    for i in range (n_samples):  
                        k = np.random.choice(mixtures, p=pi_value)
                        mean =[mu_x_point[k].detach().cpu().numpy(),mu_y_point[k].detach().cpu().numpy()]
                        cov = [[np.square(sigma_x_point[k].detach().cpu().numpy()),0],[0,np.square(sigma_y_point[k].detach().cpu().numpy())]]
                        pred = np.random.multivariate_normal(mean,cov,1)    # the last parameter is number of samples
                        mixture_mean.append(np.array(pred[0]))
                        gt_mean.append(np.array(gt_point))
                    point_pred.append(mixture_mean)
                    point_gt.append(gt_mean)
                else:
                        k = np.random.choice(mixtures, p=pi_value)
                        mean =[mu_x_point[k].detach().cpu().numpy(),mu_y_point[k].detach().cpu().numpy()]
                        cov = [[np.square(sigma_x_point[k].detach().cpu().numpy()),0],[0,np.square(sigma_y_point[k].detach().cpu().numpy())]]
                        pred = np.random.multivariate_normal(mean,cov,1)    # the last parameter is number of samples
                        point_pred.append(np.array(pred[0]))
                        point_gt.append(np.array(gt_point))
        seq_pred.append(point_pred)
        seq_gt.append(point_gt)
    seq_pred = torch.from_numpy(np.array(seq_pred)).to(device)
    seq_gt = torch.from_numpy(np.array(seq_gt)).to(device)

    return seq_pred, seq_gt   
def sampler(pi, sigma_x,sigma_y, mu_x , mu_y,gt,mixtures=5,n_samples = 1,use_mean = True,postprocess=True):
    pred_all,gt_all,pi_all = [],[],[]
    seq_pred , seq_gt = [] , []
    batch_pred , batch_gt = [] , []
    for pi_seq, sigma_x_seq,sigma_y_seq, mu_x_seq, mu_y_seq,gt_seq in zip(pi, sigma_x,sigma_y, mu_x , mu_y,gt):
        point_pred = []
        point_gt = []
        for  pi_point, sigma_x_point,sigma_y_point,mu_x_point, mu_y_point,gt_point in zip ( pi_seq, sigma_x_seq,sigma_y_seq, mu_x_seq, mu_y_seq,gt_seq):
            for i in range (n_samples):
                pi_value = pi_point.detach().cpu().numpy()   
                if use_mean:
                    #k = np.argmax(pi_value)
                    k = torch.argmax (pi_value)
                    pred =[mu_x_point[k],mu_y_point[k]]
                    # pred =[mu_x_point[k].detach().cpu().numpy(),mu_y_point[k].detach().cpu().numpy()]
                    point_pred.append(np.array(pred))
                    point_gt.append(np.array(gt_point))

                else:
                    k = np.random.choice(mixtures, p=pi_value)
                    mean =[mu_x_point[k].detach().cpu().numpy(),mu_y_point[k].detach().cpu().numpy()]
                    cov = [[np.square(sigma_x_point[k].detach().cpu().numpy()),0],[0,np.square(sigma_y_point[k].detach().cpu().numpy())]]
                    pred = np.random.multivariate_normal(mean,cov,1)    # the last parameter is number of samples
                    point_pred.append(np.array(pred[0]))
                    point_gt.append(np.array(gt_point))
        seq_pred.append(point_pred)
        seq_gt.append(point_gt)
    seq_pred = torch.from_numpy(np.array(seq_pred)).to(device)
    seq_gt = torch.from_numpy(np.array(seq_gt)).to(device)

    return seq_pred, seq_gt       
def sample_distribution(mus,sigmas,pis,gts):
    '''
    mus   - [num_batchs,bach_size, seq_length,k, features]  # k = 5 - number of different points
    sigma - [num_batchs,bach_size, seq_length,k, features]  # k = 5 - number of different points
    pis   - [num_batchs,bach_size, seq_length,k]   # pi is the weight of each Gaussian mixture
    gts   - [num_batchs,bach_size, seq_length, features]   # ground truth

    '''
    mixtures = 5 # number of mixtures
    pred_all,gt_all,pi_all = [],[],[]
    seq_pred , seq_gt = [] , []
    for mue_batch,sigma_batch,pi_batch,gt_batch in zip(mus,sigmas,pis,gts):  # iterate over batchs --> batch
        #mue_batch =  np.swapaxes(mue_batch,0,1)
        #sigma_batch =  np.swapaxes(sigma_batch,0,1)
        for mue_seq,sigma_seq,pi_seq,gt_seq in zip (mue_batch,sigma_batch,pi_batch,gt_batch):  # iterate over instances (sequences) --> sample sequence
            # mue =  np.swapaxes(mue,0,1)
            # sigma =  np.swapaxes(sigma,0,1)
            #pred_sample,gt_sample,pi_sample = [],[],[]
            point_pred = []
            point_gt = []
  
            for mue_point,sigma_point,pi_point,gt_point in zip (mue_seq,sigma_seq,pi_seq,gt_seq):



                k = np.random.choice(5, p=pi_point)

                selected_mu = mue_point[k]
                selected_cov = sigma_point[k]
                # print("selected_mu: ",selected_mu)
                # print("selected_cov: ",selected_cov)
                mean =[selected_mu[0],selected_mu[1]]
                cov = [[np.square(selected_cov[0]),0],[0,np.square(selected_cov[1])]]
                pred = np.random.multivariate_normal(mean,cov,1)    # the last parameter is number of samples
                point_pred.append(pred[0])
                point_gt.append(gt_point)
            
            seq_pred.append(point_pred)
            seq_gt.append(point_gt)



                # for mue_mix,sigma_mix,pi_mix in zip(mue_point,sigma_point,pi_point): 
                #     #cov = [[np.square(sigmax),0],[0,np.square(sigmay)]]
                #     mean =[mue_mix[0],mue_mix[1]]
                #     cov = [[np.square(sigma_mix[0]),0],[0,np.square(sigma_mix[1])]]
                #     #cov = [[sigma_mix[0],0],[0,sigma_mix[1]]]
                #     pred = np.random.multivariate_normal(mean,cov,1)    # the last parameter is number of samples
                #     point_pred.append(pred[0])
                #     point_gt.append(gt_point)
                #     #pi_sample.append(pi_mix)
                # seq_pred.append(point_pred)
                # seq_gt.append(point_gt)
                

            # pred_all.append(pred_sample)
            # gt_all.append(gt_sample)
            # pi_all.append(pi_sample)
    
    return seq_pred,seq_gt