import numpy as np
import torch
import torch.nn as nn 
import math
from torch import nn, Tensor
import pickle
from config import CFG,Args
from matplotlib import pyplot as plt
import os

# Set up 
setup = CFG()


def visualize_preds_only(src_trajs,candidate_trajs,candidate_weights):
    for src_batch,candidate_batch,candidate_weight in zip(src_trajs,candidate_trajs,candidate_weights):
            for sample_src,sample_traj,sample_weight in zip(src_batch,candidate_batch,candidate_weight):
            
                fig = plt.figure(1)	#identifies the figure 
                plt.title(" Predictions ", fontsize='16',)	#title
                plt.scatter(sample_src[:,0],sample_src[:,1],color='Black',label ='Observed trajectory')
                # plt.scatter(sample_gt[:,0],sample_gt[:,1],color='Red',label ='Ground truth')	#plot the points

                length = len(sample_traj)
                # print(sample_traj)
                clr = ['Green','Blue','Orange','yellow','brown','cyan']
                colors = clr[:length]
                # print("clr = ",length,"colors: ",colors)

                i = 0
                sum_weights = 0
                probs_list = []
                label_list = []
                

                for traj,cl,wgt in zip(sample_traj,colors,sample_weight):
                    i+=1
                    values = np.array(traj)
                    label_name = 'Pred_' + str(i) + " prob =  " +  str(round((wgt[0]*100), 2)) + " %"
                    # sum_weights += weight
                    # probs_list.append(weight)
                    # label_list.append(label_name)

                    # print("weight: ",traj[1])
                    plt.scatter(values[:,0],values[:,1],color=cl,label=label_name)
            
                # probs_list = probs_list/sum_weights
                # print("probs_list: ",probs_list)
                # pred_labels = [(label_name + str(round((prob[0]*100), 2)) + " %") for label_name,prob in zip(label_list,probs_list) ]
                # labels = ['Observed trajectory','Ground truth'] + pred_labels
        

                if(Args.dataset_name=='hotel'):
                    plt.xlim((-3, 8))
                    plt.ylim((-10, 8))
                else:
                    plt.xlim((0, 15))
                    plt.ylim((0, 15))
                # plt.legend(labels,loc="upper right")
                plt.legend()
                plt.show()

def traj_err(truth,pred,last=11):
        ade_sum = 0
        fde_sum = 0
        fde_counter=0
        ade_counter=0 
        for j in range (len(pred)): #12
    
                err = math.sqrt((pred[j][0] - truth[j][0]) ** 2 + (pred[j][1] - truth[j][1]) ** 2)
                ade_sum +=err
                if(j==last): 
                        fde_sum += err
                        fde_counter+=1
                ade_counter+=1
        return (ade_sum/ade_counter),(fde_sum/fde_counter)
def visualize_preds (src_trajs,batch_gts,candidate_trajs,candidate_weights):
    for src_batch,gt_batch,candidate_batch,candidate_weight in zip(src_trajs,batch_gts,candidate_trajs,candidate_weights):
        for sample_src,sample_gt,sample_traj,sample_weight in zip(src_batch,gt_batch,candidate_batch,candidate_weight):
        

            fig = plt.figure(1)	#identifies the figure 
            plt.title(" Predictions vs Ground truth", fontsize='16',)	#title
            plt.scatter(sample_src[:,0],sample_src[:,1],color='Black',label ='Observed trajectory')
            plt.scatter(sample_gt[:,0],sample_gt[:,1],color='Red',label ='Ground truth')	#plot the points

            length = len(sample_traj)
            # print(sample_traj)
            clr = ['Green','Blue','Orange','yellow','brown','cyan']
            colors = clr[:length]
            # print("clr = ",length,"colors: ",colors)

            i = 0
            sum_weights = 0
            probs_list = []
            label_list = []
            

            for traj,cl,wgt in zip(sample_traj,colors,sample_weight):
                i+=1
                values = np.array(traj)
                label_name = 'Pred_' + str(i) + " prob =  " +  str(round((wgt[0]*100), 2)) + " %"
                # sum_weights += weight
                # probs_list.append(weight)
                # label_list.append(label_name)

                # print("weight: ",traj[1])
                plt.scatter(values[:,0],values[:,1],color=cl,label=label_name)
        
            # probs_list = probs_list/sum_weights
            # print("probs_list: ",probs_list)
            # pred_labels = [(label_name + str(round((prob[0]*100), 2)) + " %") for label_name,prob in zip(label_list,probs_list) ]
            # labels = ['Observed trajectory','Ground truth'] + pred_labels
     

            if(Args.dataset_name=='hotel'):
                plt.xlim((-3, 8))
                plt.ylim((-10, 8))
            else:
                plt.xlim((0, 15))
                plt.ylim((0, 15))
            # plt.legend(labels,loc="upper right")
            plt.legend()
            plt.show()
def visualize_preds_old (src_trajs,batch_gts,candidate_trajs):
    for src_batch,gt_batch,candidate_batch in zip(src_trajs,batch_gts,candidate_trajs):
        for sample_src,sample_gt,sample_traj in zip(src_batch,gt_batch,candidate_batch):
        

            fig = plt.figure(1)	#identifies the figure 
            plt.title(" Predictions vs Ground truth", fontsize='16')	#title
            plt.scatter(sample_src[:,0],sample_src[:,1],color='Black')
            plt.scatter(sample_gt[:,0],sample_gt[:,1],color='Red')	#plot the points

            length = len(sample_traj)
            # print(sample_traj)
            clr = ['Green','Blue','Orange','yellow','brown','cyan']
            colors = clr[:length]
            # print("clr = ",length,"colors: ",colors)

            i = 0
            sum_weights = 0
            probs_list = []
            label_list = []
            

            for traj,cl in zip(sample_traj,colors):
                i+=1
                values = np.array(traj[0])
                weight = traj[1]

                label_name = 'Pred_' + str(i) + " prob =  "
                sum_weights += weight
                probs_list.append(weight)
                label_list.append(label_name)

                # print("weight: ",traj[1])
                plt.scatter(values[:,0],values[:,1],color=cl)
        
            probs_list = probs_list/sum_weights
            print("probs_list: ",probs_list)
            pred_labels = [(label_name + str(round((prob*100), 2)) + " %") for label_name,prob in zip(label_list,probs_list) ]
            labels = ['Observed trajectory','Ground truth'] + pred_labels
     

            if(Args.dataset_name=='hotel'):
                plt.xlim((-3, 8))
                plt.ylim((-10, 8))
            else:
                plt.xlim((0, 15))
                plt.ylim((0, 15))
            plt.legend(labels,loc="upper right")
            plt.show()
def FDE_Single(pred,truth): 
    counter=0
    sum=0
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()

    for ai,bi in zip(pred,truth):
        # print(ai,bi)
        dist = np.linalg.norm(ai-bi)
        sum+=dist
        counter+=1
    return (sum/counter)
def FDE_Single_GMM(pred,truth): 
    counter=0
    sum=0
    indexs = []
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()

    for pred_batch, truth_batch in zip(pred,truth):
        dist =  np.linalg.norm(pred_batch[0]-truth_batch[0])
        index = 0
        for i ,(ai,bi) in enumerate (zip(pred_batch,truth_batch)):
            # print(ai,bi)
            dist_cal = np.linalg.norm(ai-bi)
            if dist_cal <= dist:
                dist = dist_cal
                index = i
        sum+=dist
        indexs.append(index)
        counter+=1
    return (sum/counter),indexs
def show_error_single(prediction_train,prediction_test,train_target,test_target):
                # # Show error rate
        print("AVERAGE DISTANCE BETWEEN FIRST AND LAST POINT Train: ",prediction_displacement(train_target))
        print("AVERAGE DISTANCE BETWEEN FIRST AND LAST POINT Test: ",prediction_displacement(test_target))
        print("////////////////////////////////////////////////")
        #average_displacement_error
        print("ADE ERROR RATE TEST: ", FDE_Single(prediction_test,test_target))
        #average_displacement_error
        print("ADE ERROR RATE TRAIN: ", FDE_Single(prediction_train,train_target))
        print("//////////////////////////////////////////")
        #Final_displacement_error
        print("FDE ERROR RATE TEST: ", FDE_Single(prediction_test,test_target))
        print("FDE ERROR RATE TRAIN: ", FDE_Single(prediction_train,train_target))

# Predict error
# Average Displacement error

def ADE_old(pred,truth): 
    counter=0
    sum=0
    for i in range(len(pred)):
        half=int(len(pred[i])/2)
        for j in range (half):

            a = np.array((pred[i][j] , pred[i][half+j]))
            b = np.array((truth.iloc[i][j] , truth.iloc[i][half+j]))

            dist = np.linalg.norm(a-b)
            sum+=dist
            counter+=1
            #print("Distance between",a," and ",b," is: ",dist)

    return (sum/counter)
def prediction_displacement_double(pred): 
          
        sum=0
        counter=0        
        pred=np.array(pred)
        last_index= (len(pred[0])-1)
        
        for i in range(len(pred)):
                
                    a = np.array((pred[i][0][0] , pred[i][0][1]))
                    b = np.array((pred[i][last_index][0] , pred[i][last_index][1]))

                    dist = np.linalg.norm(a-b)
                    sum+=dist
                    counter+=1
        return (sum/counter)

def prediction_displacement(pred): 
    counter=0
    sum=0
    pred=np.array(pred)
    for i in range(len(pred)):
        half=int(len(pred[i])/2)
        last=(len(pred[i]) - 1)

        a = np.array((pred[i][0] , pred[i][half]))
        b = np.array((pred[i][half-1] , pred[i][last]))

        dist = np.linalg.norm(a-b)
        sum+=dist
        counter+=1
        #print("FDE Distance between",a," and ",b," is: ",dist)
            
    return (sum/counter)
def FDE_old(pred,truth): 
    counter=0
    sum=0
    for i in range(len(pred)):
        half=int(len(pred[i])/2)
        last=(len(pred[i]) - 1)

        a = np.array((pred[i][half-1] , pred[i][last]))
        b = np.array((truth.iloc[i][half-1] , truth.iloc[i][last]))

        dist = np.linalg.norm(a-b)
        sum+=dist
        counter+=1
        #print("FDE Distance between",a," and ",b," is: ",dist)
            
    return (sum/counter)
def show_error_double(prediction_train,prediction_test,train_target,test_target):
                # # Show error rate
        print("AVERAGE DISTANCE BETWEEN FIRST AND LAST POINT Train: ",prediction_displacement_double(train_target))
        print("AVERAGE DISTANCE BETWEEN FIRST AND LAST POINT Test: ",prediction_displacement_double(test_target))
             
        #average_displacement_error
        print("ADE ERROR RATE TEST: ", ADE_double_coordinates(prediction_test,test_target))
        #average_displacement_error
        print("ADE ERROR RATE TRAIN: ", ADE_double_coordinates(prediction_train,train_target))
        print("//////////////////////////////////////////")
        #Final_displacement_error
        print("FDE ERROR RATE TEST: ", FDE_double_coordinates(prediction_test,test_target))
        print("FDE ERROR RATE TRAIN: ", FDE_double_coordinates(prediction_train,train_target))
def FDE(pred,truth):
        sum=0
        counter=0
        
        pred=np.array(pred)
        truth=np.array(truth)
        last_index= (len(pred[0])-1)

        for i in range(len(pred)):
                
            a = np.array((pred[i][last_index][0] , pred[i][last_index][1]))
            b = np.array((truth[i][last_index][0] , truth[i][last_index][1]))

            dist = np.linalg.norm(a-b)
            sum+=dist
            counter+=1

        return (sum/counter)
def ADE(pred,truth):
          
        sum=0
        counter=0
        
        pred=np.array(pred)
        truth=np.array(truth)

        for i in range(len(pred)):
                
                for j in range (len(pred[i])):

                    a = np.array((pred[i][j][0] , pred[i][j][1]))
                    b = np.array((truth[i][j][0] , truth[i][j][1]))

                    dist = np.linalg.norm(a-b)
                    sum+=dist
                    counter+=1
        return (sum/counter)

def Error(gts,preds,type='ADE'):
    sum,counter,max,m_index = 0,0,0,0
    min = 10
    for i , (gts_batch,preds_batch) in enumerate(zip(gts,preds)):
        err = ADE(gts_batch,preds_batch) if (type=='ADE') else FDE(gts_batch,preds_batch)
        sum  += err
        counter+=1
        if err > max:
            max = err
            m_index = i
        if err < min:
            min = err
        #print(err)
    avg = sum/counter
    #print(f"Average Error: {avg}\nMax Error per batch: {max} at index:  {m_index}\nMin Error per batch: {min}")
    return avg,max,min

def load_data():
    with open('train_input.pickle', 'rb') as data:
     train_input = pickle.load(data)
    with open('validate_input.pickle', 'rb') as data:
         validate_input= pickle.load(data)
    with open('test_input.pickle', 'rb') as data:
         test_input = pickle.load(data)
    with open('train_target.pickle', 'rb') as data:
          train_target = pickle.load(data)
    with open('validate_target.pickle', 'rb') as data:
          validate_target = pickle.load(data)
    with open('test_target.pickle', 'rb') as data:
         test_target = pickle.load(data)
    return torch.tensor(train_input,dtype=torch.float32), torch.tensor(validate_input,dtype=torch.float32), torch.tensor(test_input,dtype=torch.float32), torch.tensor(train_target,dtype=torch.float32), torch.tensor(validate_target,dtype=torch.float32), torch.tensor(test_target,dtype=torch.float32)

def create_tgt(src,target,token_zero = False,zero_feed=False,seq_length = 8):
    # If token_zero is true then the first token is changed to zero 
    if zero_feed:
        return torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(src.shape[0], seq_length, 1)
    if token_zero:
        sos_token = torch.Tensor([[[0, 0]]]) # special token of start of sentence
        sos_token = sos_token.repeat(target.shape[0],1,1) # Should be same size as tensor
        return torch.cat((sos_token[:,:1,:],target[:,:-1,:]), dim = 1)
    else:
        sos_token = src[:,-1,:]
        sos_token = torch.unsqueeze(sos_token, 1)
        return torch.cat((sos_token,target[:,:-1,:]), dim = 1)

def Err(truth,pred,last=11):
        ade_sum = 0
        fde_sum = 0
        fde_counter=0
        ade_counter=0

        for i in range(len(pred)):  #100
                
                for j in range (len(pred[i])): #12
          
                        err = math.sqrt((pred[i][j][0] - truth[i][j][0]) ** 2 + (pred[i][j][1] - truth[i][j][1]) ** 2)
                        ade_sum +=err
                        if(j==last): 
                                fde_sum += err
                                fde_counter+=1
                        ade_counter+=1

        return (ade_sum/ade_counter),(fde_sum/fde_counter)

'''
A wrapper class for scheduled optimizer 
source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
'''
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        #print(self.n_warmup_steps)
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0

def generate_square_mask(dim_trg: int, dim_src: int, mask_type: str) -> torch.Tensor:
    """
    This is used to create causality constriant into the decoder input.
    This technique helps prevent the model from attending the future positions 
    in the input/target sequence when generating output sequence. This is done by masking out
    certain positions in the input/target sequence so that they can not be attended.

    src: (S,E)
    tgt: (T,E)
    src_mask: (S,S)
    tgt_mask: (T,T)
    memory_mask: (T,S)

    where S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number


    Generates a square mask of -inf or 0, based on mask_type.
    The mask can be used for src or tgt masking.
    For src masking, mask_type should be "src"
    For tgt masking, mask_type should be "tgt"

    
    Args:
        seq_len: int, the sequence length for which the mask will be generated
        mask_type: str, should be either "src" or "tgt"
    
    Returns:
        A Tensor of shape [seq_len, seq_len]
    """
    mask = torch.ones(dim_trg, dim_trg)* float('-inf')
    if mask_type == "src":
        '''
        src_mask [Tx, Tx] = [S, S] – the additive mask for the src sequence (optional). 
        This is applied when doing atten_src + src_mask. I'm not sure of an example input - 
        the typical use is to add -inf so one could mask the src_attention that way if desired. 
        If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged. 
        If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged.
        If a FloatTensor is provided, it will be added to the attention weight.
        
        '''
        mask = torch.triu(mask, diagonal=1)
    elif mask_type == "tgt":
        mask = torch.triu(mask, diagonal=1)
        #mask = torch.tril(mask)
    elif mask_type == "memory":
        mask = torch.ones(dim_trg, dim_src)* float('-inf')
        mask = torch.triu(mask, diagonal=1)
    return mask
# Pytorch code edited to include batch mode (first or not)
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        if batch_first: 
            pe = torch.zeros(1,max_len, d_model)
            pe[0,:, 0::2] = torch.sin(position * div_term)
            pe[0,:, 1::2] = torch.cos(position * div_term)
        else: 
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] 
            x: Tensor, shape [batch_size, seq_len, embedding_dim]batch first
        """
        #print("pe[:,:x.size(1),:] shape: ",self.pe.shape)
        x = x + self.pe[:,:x.size(1),:] if self.batch_first else x + self.pe[:x.size(0)]

        return self.dropout(x)
class early_save_stop:
    def __init__(self, tolerance=10, min_delta=0.02 , last_loss = 10.0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.last_validation = last_loss
        self.best_val_batch_mad = 10
        self.best_val_batch_fad = 10
        self.stop = False
        self.previous_epoch = 0

    def __call__(self, model,train_loss, validation_loss,epoch,val_batch_mad,val_batch_fad):
        #(validation_loss < self.last_validation) and 

        delta_ADE = self.best_val_batch_mad - val_batch_mad
        delta_FDE = self.best_val_batch_fad - val_batch_fad
        delta = delta_ADE + delta_FDE
        #(val_batch_mad< self.best_val_batch_mad) and

        if ( (val_batch_fad < self.best_val_batch_fad) and (delta_ADE > 0 and delta_FDE > 0)):
        #if (delta > 0):
            print("Saving Model!")
            print("Change in validation loss: ",(self.last_validation - validation_loss))
            print("Change in Validation ADE: ",(delta_ADE))
            print("Change in Validation FDE: ",(delta_FDE))
            
            if(self.previous_epoch > 0):
                old_path =  CFG.PATH +str(self.previous_epoch)+'.pt'
                old_path_pth =  CFG.PATH +str(self.previous_epoch)+'.pth'
                os.remove(old_path)
                os.remove(old_path_pth)
            torch.save(model, CFG.PATH +str(epoch)+'.pt')
            torch.save(model.state_dict(), CFG.PATH +str(epoch)+'.pth')
            print("Validation: ",validation_loss)
            print("Model saved!")

            # update
            self.counter = 0
            self.previous_epoch = epoch
            self.last_validation = validation_loss
            self.best_val_batch_mad = val_batch_mad
            self.best_val_batch_fad = val_batch_fad
        else:
            if (abs(validation_loss - train_loss)) > self.min_delta:
                self.counter +=1
                if self.counter >= self.tolerance:  
                    self.stop = True

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x, y , tgt=None):
        'Initialization'
        self.x = x
        self.y = y
        self.tgt = tgt
        
      #   self.start = 0

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)
      
  def base(self,index):
      return self.x[index][0]
  
  def input_shape(self):
        return self.x.shape
  def target_shape(self):
        return self.y.shape

  def __getitem__(self, index):
        # Normalize
        # Load data and get label
        device = CFG.device
        #print(f"Using {device} device")
        
      #   self.start = self.x[index][0]
        X = self.x[index].to(device) #- self.x[index][7].expand_as(self.x[index]).to(device)
      
        Y = self.y[index].to(device) #- self.x[index][7].expand_as(self.y[index]).to(device)

        tgt = self.tgt[index].to(device) #tgt for decoder



        # X, y,start ---> start is the first x and y of the sample

        return X, Y, tgt


        

