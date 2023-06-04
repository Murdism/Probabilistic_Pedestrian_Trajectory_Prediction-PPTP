import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from torch.autograd import Variable # storing data while learning
from loss import mdn_loss_fn,pairwise_distance,Mean_squared_distance
from config import CFG,Args
from utils import create_tgt,generate_square_mask,ADE,subsequent_mask,early_save_stop
from baselineUtils import distance_metrics
from tqdm import tqdm
import torch.nn.functional as F  
from test import test_mdn
from mdn import sampler,sample_mean


device_train = CFG.device
batch_size = CFG.batch_size
# print(f"Using {device} device")

def train_attn_mdn(train_dl,val_dl,test_dl, model, optim,add_features = False,mixtures = 8,device=device_train, epochs=20,enc_seq = 8,real_coords =False,dec_seq=12,post_process= True,normalized ='True', mode='feed',loss_mode ='combined',mean=0,std=0):
    early_stop = early_save_stop()
    encoder_seq_len = enc_seq
    decoder_seq_len = dec_seq  
    num_batches = len(train_dl)
    num_batches_val = len(val_dl)

    loss_list,loss_eval,all_mad,all_fad = [],[],[],[]
    val_mad,val_fad,test_mad,test_fad = [],[],[],[]
    print('Training Setting:... ')
    print(f"Train Num of batchs {num_batches}\nloss: {loss_mode}\nData Normalized: {normalized}")
    for epoch in range (epochs):
        gts_train, preds_train = [] , []
        epoch_loss,epoch_val_loss = 0 , 0
        val_epoch_mad,val_epoch_fad = [],[]
        load_train = tqdm(train_dl)
        # load_val = tqdm(val_dl)

        model.train()
        for id_b,batch in enumerate(load_train):

            load_train.set_description(f"Epoch: {epoch+1} / {epochs}")

            #src_mask = generate_square_mask(dim_trg = encoder_seq_len ,dim_src = decoder_seq_len,mask_type="memory").to(device)

            if (normalized):
                
                inp=(batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
                target=(batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
                y = (batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)


            else:
                inp = batch['src'][:,1:,2:4].to(device).to(device)
                target = batch['trg'][:,:-1,2:4].to(device).to(device)
                y = batch['trg'][:, :, 2:4].to(device)
            if real_coords:
                 y = (batch['trg'][:, :, 0:2]).to(device)
            if (add_features):
                
                input_c = torch.sqrt(torch.square(batch['src'][:,1:,2].to(device)) + torch.square(batch['src'][:,1:,3].to(device))).unsqueeze(-1)
                #input_d = (batch['src'][:,1:,3].to(device) / batch['src'][:,1:,2].to(device)).unsqueeze(-1)
                input = torch.cat((inp,input_c),-1)
                # input = torch.cat((input_temp,input_d),-1)
                # print("input.shape: ", input.shape)
            else:
                input = inp
                target = target


            target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
            target=torch.cat((target,target_c),-1)
            tgt = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],12,1).to(device)

            tgt_mask = generate_square_mask(dim_trg = decoder_seq_len ,dim_src = encoder_seq_len, mask_type="tgt").to(device)
            #tgt_mask = subsequent_mask(decoder_seq_len).to(device)

            optim.zero_grad()
            pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = model(input,tgt,tgt_mask = tgt_mask)
            mus = torch.cat((mu_x.unsqueeze(-1),mu_y.unsqueeze(-1)),-1)
            sigmas = torch.cat((sigma_x.unsqueeze(-1),sigma_y.unsqueeze(-1)),-1)
            # print("MUS: ",mus.shape, "\nPI: ",pi.shape)
            # print("MUS: ",mus[0][0], "\nPI: ",pi[0][0])

            batch_pred = sample_mean(pi,sigmas,mus)

            if loss_mode=='pair_wise':
                # remeber decoder_out is not being optimized but sigmas and mus
                loss = F.pairwise_distance(decoder_out[:, :,0:2].contiguous().view(-1, 2),((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(decoder_out[:,:,2]))
               
            elif(loss_mode=='msq'):
                # remeber decoder_out is not being optimized but sigmas and mus
                pred = decoder_out[:, :,0:2].contiguous().view(-1, 2)
                y = ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device) 
                loss = nn.MSELoss(pred,y)
            elif(loss_mode=='mdn'):
                #loss_mdn 
                train_loss = mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,y,mixtures,device)
                
            elif(loss_mode=='combined'):

                loss_mdn = mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,y,mixtures,device)
                msq = Mean_squared_distance(y.contiguous().view(-1, 2).to(device),batch_pred.to(device).contiguous().view(-1, 2))
                train_loss =  model.mdn_weight * loss_mdn + (1- model.mdn_weight)*msq
                #print("loss_mdn: ",loss_mdn,"   loss_mdn: ",msq ," after: ",(model.mdn_weight * loss_mdn),(loss_mdn + (1- model.mdn_weight)*msq))
                
    
            
            #print("shape: ",batch_pred.contiguous().view(-1, 2).shape)
            train_loss.backward()
            # Update the learning rate schedule
            optim.step_and_update_lr()

            epoch_loss += train_loss.item()
        
        avg_loss_epoch = epoch_loss / num_batches
        loss_list.append(avg_loss_epoch)
        
        lr = optim._optimizer.param_groups[0]['lr']

        


        ## EVALUATE in validation

        with torch.no_grad():
            model.eval()
            gts_ev,preds_ev,src_ev = [], [] ,[]

            for id_e, val_batch in enumerate(val_dl):
                #load_val.set_description(f"Epoch: {epoch+1} / {epochs}")
                src_ev.append(val_batch['src'])
                gts_ev.append(val_batch['trg'][:, :, 0:2])
                batch_gt_val = val_batch['trg'][:, :, 0:2].to(device)

                if(normalized):
                    inp_val=(val_batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
                    target_val=(val_batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
                    y_val = (val_batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)

                    # input_valc = torch.sqrt(torch.square(val_batch['src'][:,1:,2].to(device)) + torch.square(val_batch['src'][:,1:,3].to(device))).unsqueeze(-1)

                    # input_val = torch.cat((inp_val,input_valc),-1)
                else:
                    inp_val = val_batch['src'][:,1:,2:4].to(device)
                    target_val = val_batch['trg'][:,:-1,2:4].to(device)
                    y_val = val_batch['trg'][:, :, 2:4].to(device)
                    batch_gt_val = val_batch['trg'][:, :, 0:2].to(device)
                
                if real_coords:
                    y_val = (val_batch['trg'][:, :, 0:2]).to(device)

                if (add_features):                

                    input_valc = torch.sqrt(torch.square(val_batch['src'][:,1:,2].to(device)) + torch.square(val_batch['src'][:,1:,3].to(device))).unsqueeze(-1)
                    input_val = torch.cat((inp_val,input_valc),-1)
                else:
                    input_val = inp_val
                    target_val = target_val


                tgt_val = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(target_val.shape[0],12,1).to(device)

               
                
                tgt_val_mask = generate_square_mask(dim_trg = decoder_seq_len ,dim_src = encoder_seq_len, mask_type="tgt").to(device)

                pi_val, sigma_x_val,sigma_y_val, mu_x_val , mu_y_val,decoder_out = model(input_val,tgt_val,tgt_mask = tgt_val_mask)

                mus_val = torch.cat((mu_x_val.unsqueeze(-1),mu_y_val.unsqueeze(-1)),-1)
                sigmas_val = torch.cat((sigma_x_val.unsqueeze(-1),sigma_y_val.unsqueeze(-1)),-1)

                batch_pred_val = sample_mean(pi_val,sigmas_val,mus_val)

                #params = [pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out]

                if(loss_mode=='mdn'):
                    #loss_mdn 
                    loss_val = mdn_loss_fn(pi_val, sigma_x_val,sigma_y_val, mu_x_val , mu_y_val,y_val,mixtures,device)
                    
                elif(loss_mode=='combined'):
                    loss_val_mdn = mdn_loss_fn(pi_val, sigma_x_val,sigma_y_val, mu_x_val , mu_y_val,y_val,mixtures,device)
                    msq_val = Mean_squared_distance(y_val.contiguous().view(-1, 2).to(device),batch_pred_val.contiguous().view(-1, 2).to(device))

                    loss_val =  model.mdn_weight * loss_val_mdn + (1- model.mdn_weight)*msq_val
                    #print("loss_mdn: ",loss_mdn,"   loss_mdn: ",msq ," after: ",(model.mdn_weight * loss_mdn),(loss_mdn + (1- model.mdn_weight)*msq))

                epoch_val_loss += loss_val.item()

                batch_gt_val = batch_gt_val.detach().cpu().numpy()
            
                if (post_process and normalized):
                #print("TRUE")
                    batch_pred_val = (batch_pred_val[:, :,:] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + val_batch[
                                                                                                                'src'][
                                                                                                            :, -1:,
                                                                                                            0:2].cpu().numpy()
                    batch_gt_val = val_batch['trg'][:, :, 0:2].to(device).cpu().numpy()
                else:
                    batch_pred_val = batch_pred_val.detach().cpu().numpy()
                    batch_pred_val = batch_pred_val[:, :,:].cumsum(1) + val_batch['src'][:, -1:,0:2].cpu().numpy()
                # calcualte error:
 
                mad, fad, errs = distance_metrics(batch_gt_val, batch_pred_val)
                val_epoch_mad.append(mad)
                val_epoch_fad.append(fad)

        
        avg_loss_epoch_val = epoch_val_loss / num_batches_val
        sum(val_epoch_mad)
        val_batch_mad = (sum(val_epoch_mad)/num_batches_val)
        val_batch_fad = (sum(val_epoch_fad)/num_batches_val)
        val_mad.append(val_batch_mad)
        val_fad.append(val_batch_fad)
        loss_eval.append(avg_loss_epoch_val)

        # save and stop model
        early_stop(model,avg_loss_epoch, avg_loss_epoch_val,epoch+1,val_batch_mad,val_batch_fad)
        print(f"Train loss:{avg_loss_epoch:.4f} mdn weight: {model.mdn_weight:.3f}") #mdn weighte: {model.mdn_weight}
        print(f"Eval loss: {avg_loss_epoch_val:.4f}")
        print(f"Eval val_batch_mad: {val_batch_mad:.4f}")
        print(f"Eval val_batch_fad: {val_batch_fad:.4f}")
        print(f"Learning Rate: {lr:.5f}")
        
        #mad_test , fad_test,_,_,mdn_results,avg_mad,avg_fad= test_mdn(test_dl, model,device,add_features = add_features,mixtures=mixtures,enc_seq = 8,dec_seq=12, mode='feed',loss_mode ='mdn',mean=mean,std=std)
        # test_mad.append(avg_mad)
        # test_fad.append(avg_fad)
        # Test the model
        if Args.show_test:
            print('----- Test -----')
            batch_preds,batch_gts,avg_mad,avg_fad,candidate_trajs,candidate_weights,best_candiates,src_trajs = test_mdn(test_dl, model,device,add_features = add_features,mixtures=mixtures,enc_seq = 8,dec_seq=12, mode='feed',loss_mode ='mdn',mean=mean,std=std)
            print('----- END -----')

        # if (early_stop.stop):
        #     print("Early stopping activated!")
        #     break

        

    print(f"Epoch {epoch+1} Train loss: {avg_loss_epoch}")
    print(f"Epoch {epoch+1} Eval loss: {avg_loss_epoch_val}")
    return loss_list, loss_eval,val_mad,val_fad#all_mad,all_fad,loss_list






# def train_transformer_eth(train_dl,val_dl, model,optim,device=device_train,epochs=20,enc_seq = 8,dec_seq=12, mode='feed',loss_mode ='pair_wise',mean=0,std=0):
    

#     loss_fn = nn.MSELoss()
#     loss_list,all_mad,all_fad = [],[],[]
#     for epoch in range (epochs):
#         total_loss = 0
#         num_batches = len(train_dl)
#         total_loss = 0
#         model.train()
#         encoder_seq_len = enc_seq
#         decoder_seq_len = dec_seq
#         gts_train, preds_train = [] , []
#         load_train = tqdm(train_dl)
#         load_val = tqdm(val_dl)
#         epoch_loss = 0
#         # Perform a linear warm-up for the first 5 epochs
#         # if epoch < 5:
#         #     optimizer.param_groups[0]['lr'] = 0.0002 + (epoch / 5) * (0.001 - 0.0002)
#         model.train()
#         for id_b,batch in enumerate(load_train):

            
            
#             load_train.set_description(f"Epoch: {epoch+1} / {epochs}")


#             #src_mask = generate_square_mask(dim_trg = encoder_seq_len ,dim_src = decoder_seq_len,mask_type="memory").to(device)

#             inp=(batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
#             target=(batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
#             target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
#             target=torch.cat((target,target_c),-1)
#             start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)

#             tgt = torch.cat((start_of_seq, target), 1)

#             #src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
#             #trg_att=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)

#             tgt_mask = generate_square_mask(dim_trg = decoder_seq_len ,dim_src = encoder_seq_len, mask_type="tgt").to(device)
#             #tgt_mask = subsequent_mask(decoder_seq_len).to(device)

#             optim.zero_grad()
#             pred = model(inp,tgt,tgt_mask = tgt_mask)

#             dec_pred = (pred[:, 1:, 0:2] * std.to(device) + mean.to(device)).detach().cpu().numpy().cumsum(1) + batch[
#                                                                                                                     'src'][                                                                                                         :, -1:,
#                                                                                                         0:2].cpu().numpy()

#             #pred = pred[:, :,0:2].contiguous().view(-1, 2)

#             #print(f"pred: {type(pred)} \nY : {type(y)}")
#             #y = ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device))#.contiguous().view(-1, 2).to(device).mean() + torch.mean(torch.abs(pred[:,:,2]))
#             y = (batch['trg'][:, :, 2:4].to(device)).contiguous().view(-1, 2)
#             if loss_mode=='pair_wise':
#                 #loss = pairwise_distance(pred,y)
#                 loss = F.pairwise_distance(pred[:, :,0:2].contiguous().view(-1, 2),
#                                        ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(pred[:,:,2]))
#                 #loss = torch.mean(F.pairwise_distance(pred,y))
#             else:
#                 loss = loss_fn(pred[:, :,0:2].contiguous().view(-1, 2),y)
            
#             loss.backward()
#             optim.step_and_update_lr()

#             epoch_loss += loss.item()


#             gts_train.append(batch['trg'][:, :, 0:2]) 
#             preds_train.append(dec_pred)

#         # Update the learning rate schedule
#         #scheduler.step()
#         avg_loss_epoch = epoch_loss / num_batches
#         loss_list.append(avg_loss_epoch)
        
#         lr = optim._optimizer.param_groups[0]['lr']

#         print(f"Epoch {epoch+1} average train loss: {avg_loss_epoch:.4f} Learning rate: {lr:.5f}")


#         ### EVALUATE in validation

#         with torch.no_grad():
#             model.eval()
#             gts_ev,preds_ev,src_ev = [], [] ,[]

#             for id_e, batch in enumerate(load_val):
#                 load_val.set_description(f"Epoch: {epoch+1} / {epochs}")
#                 src_ev.append(batch['src'])
#                 gts_ev.append(batch['trg'][:, :, 0:2])

#                 inp=(batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
#                 target=(batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
#                 target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
#                 target=torch.cat((target,target_c),-1)
#                 start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0],1,1).to(device)

#                 tgt = start_of_seq

#                 for i in range(decoder_seq_len):
#                     tgt_mask = generate_square_mask(dim_trg = tgt.shape[1] ,dim_src = tgt.shape[1], mask_type="tgt").to(device)
#                     #tgt_mask = subsequent_mask(tgt.shape[1]).to(device)
#                     pred = model(inp,tgt,tgt_mask = tgt_mask)
#                     tgt=torch.cat((tgt,pred[:,-1:,:]),1)
#                 preds_eval=(tgt[:,1:,0:2]*std.to(device)+mean.to(device)).detach().cpu().numpy().cumsum(1)+batch['src'][:,-1:,0:2].detach().cpu().numpy()
#                 preds_ev.append(preds_eval)
 
#             gts_val = np.concatenate(gts_ev, 0)  
#             preds_val = np.concatenate(preds_ev, 0)

#             mad, fad, errs = distance_metrics(gts_val, preds_val)
#             all_mad.append(mad)
#             all_fad.append(fad)
#             print('Eval/MAD', mad)
#             print('Eval/FAD', fad)
            
#     print(f"Epoch {epoch+1} train loss: {avg_loss_epoch}")
#     return all_mad,all_fad,loss_list

# def train_transformer(data_loader, model, optimizer,device=device_train,epochs=20,enc_seq = 8,dec_seq=8, mode='feed',loss_mode ='pair_wise'):
    

#     loss_fn = nn.MSELoss()
#     loss_list = []
#     for i in range (epochs):
#         total_loss = 0
#         num_batches = len(data_loader)
#         total_loss = 0
#         model.train()
#         encoder_seq_len = enc_seq
#         decoder_seq_len = dec_seq
#         gts, preds = [] , []
#         load = tqdm(data_loader)
#         epoch_loss = 0
#         for X, y, tgt in load:
#             # print(X.shape)
#             #print("INput: ",X)
#             load.set_description(f"Epoch: {i+1} / {epochs}")
#             x_variable = Variable(X,requires_grad=True).to(device)
#             tgt = tgt.to(device)
#             y = y.to(device)

#             src_mask = generate_square_mask(dim_trg = encoder_seq_len ,dim_src = decoder_seq_len,mask_type="memory").to(device)
#             tgt_mask = generate_square_mask(dim_trg = encoder_seq_len ,dim_src = decoder_seq_len, mask_type="tgt").to(device)

#             pred = model(x_variable,tgt,src_mask,tgt_mask)

#             #print(f"pred: {type(pred)} \nY : {type(y)}")
#             if loss_mode=='pair_wise':
#                 loss = pairwise_distance(pred,y)
#             else:
#                 loss = loss_fn(pred,y)
            
#             # err = error(decoder_out,y)
#             # print("decoder_out: ",decoder_out[0],"\ny: ",y[0])

#             # loss = Variable(loss, requires_grad = True)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
            


#             pred = pred.detach().cpu().numpy()
#             gt = (y.detach().cpu().numpy())#.tolist()

#             gts.append(np.array(gt)) 
#             preds.append(pred)

#         avg_loss_epoch = epoch_loss / num_batches
        
#         loss_list.append(avg_loss_epoch)

        
#         #load.set_postfix({'avg_loss': avg_loss})
#         if (i+1)%5 == 0:
#             print(f"Epoch {i+1} average train loss: {avg_loss_epoch}")

#     print(f"Epoch {i+1} train loss: {avg_loss_epoch}")
#     return gts,preds,loss_list

# def train(data_loader, model, optimizer,device=device_train,epochs=20,enc_seq = 8,dec_seq=8):
#     ''' 
#         sigma:  (num_samples,num_mixtures,2,2) 
#         pi:     (num_samples,num_mixtures)
#         mue:    (num_samples,num_mixtures,2)

#         The last parameter '2' represents x and y  
#     '''
#     num_batches = len(data_loader)
#     total_loss = 0
#     model.train()
#     encoder_seq_len = enc_seq
#     decoder_seq_len = dec_seq
    
#     for i in range (epochs):
#         total_loss = 0
#         for X, y, tgt in data_loader:
#             # print(X.shape)
#             #print("INput: ",X)
#             x_variable = Variable(X,requires_grad=True).to(device)
#             tgt = tgt.to(device)
#             #X = X.to(device)
#             #y = y.to(device)
#             # Generate Masks 

#             # src_mask = generate_square_mask(seq_len=encoder_seq_len, mask_type="src").to(device)
#             # tgt_mask = generate_square_mask(seq_len=decoder_seq_len, mask_type="tgt").to(device)

#             src_mask = generate_square_mask(dim_trg = encoder_seq_len ,dim_src = decoder_seq_len,mask_type="memory").to(device)
#             tgt_mask = generate_square_mask(dim_trg = encoder_seq_len ,dim_src = decoder_seq_len, mask_type="tgt").to(device)

#             pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = model(x_variable,tgt,src_mask,tgt_mask)
#             #print("outputs: ",pi, sigma_x,sigma_y, mu_x , mu_y)
#             #pi_variable, sigma_variable, mu_variable = model(x_variable)
#             #print(f"sigma_variable{sigma_variable.shape}")
#             loss = mdn_loss_fn( pi, sigma_x,sigma_y, mu_x , mu_y, y)
#             #err = error(decoder_out,y)
#             print("decoder_out: ",decoder_out[0],"\ny: ",y[0])

#             # loss = Variable(loss, requires_grad = True)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / num_batches
#         if (i+1)%5 == 0:
#             print(f"Epoch {i+1} train loss: {avg_loss}")
#     print(f"Epoch {i+1} train loss: {avg_loss}")




# def train_attn_mdn(train_dl,val_dl,test_dl, model, optim,add_features = False,mixtures = 3,device=device_train, epochs=20,enc_seq = 8,dec_seq=12,post_process= True,normalized ='True', mode='feed',loss_mode ='mdn',mean=0,std=0):
#     early_stop = early_save_stop()
#     encoder_seq_len = enc_seq
#     decoder_seq_len = dec_seq  
#     num_batches = len(train_dl)
#     num_batches_val = len(val_dl)

#     loss_list,loss_eval,all_mad,all_fad = [],[],[],[]
#     val_mad,val_fad,test_mad,test_fad = [],[],[],[]
#     print('Training Setting:... ')
#     print(f"Train Num of batchs {num_batches}\nloss: {loss_mode}\nData Normalized: {normalized}")
#     for epoch in range (epochs):
#         gts_train, preds_train = [] , []
#         epoch_loss,epoch_val_loss = 0 , 0
#         val_epoch_mad,val_epoch_fad = [],[]
#         load_train = tqdm(train_dl)
#         # load_val = tqdm(val_dl)

#         model.train()
#         for id_b,batch in enumerate(load_train):

#             load_train.set_description(f"Epoch: {epoch+1} / {epochs}")

#             #src_mask = generate_square_mask(dim_trg = encoder_seq_len ,dim_src = decoder_seq_len,mask_type="memory").to(device)

#             if (normalized):
                
#                 inp=(batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
#                 target=(batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)


#             else:
#                 inp=(batch['src'][:,1:,2:4].to(device).to(device))
#                 target=(batch['trg'][:,:-1,2:4].to(device).to(device))
            
#             if (add_features):
                
#                 input_c = torch.sqrt(torch.square(batch['src'][:,1:,2].to(device)) + torch.square(batch['src'][:,1:,3].to(device))).unsqueeze(-1)
#                 #input_d = (batch['src'][:,1:,3].to(device) / batch['src'][:,1:,2].to(device)).unsqueeze(-1)
#                 input = torch.cat((inp,input_c),-1)
#                 # input = torch.cat((input_temp,input_d),-1)
#                 # print("input.shape: ", input.shape)
#             else:
#                 input = inp
#                 target = target


#             target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
#             target=torch.cat((target,target_c),-1)
#             # start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)
#             # end_of_seq = torch.Tensor([0, 0, 0]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],11,1).to(device)
#             # tgt = torch.cat((start_of_seq, end_of_seq), 1)

#             #tgt = torch.cat((start_of_seq, target), 1)
            

#             # print("This is target shape: ",tgt.shape,'\ntgt: ',tgt[0])

#             #src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
#             #trg_att=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)
#             tgt = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],12,1).to(device)

#             tgt_mask = generate_square_mask(dim_trg = decoder_seq_len ,dim_src = encoder_seq_len, mask_type="tgt").to(device)
#             #tgt_mask = subsequent_mask(decoder_seq_len).to(device)

#             optim.zero_grad()
#             pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = model(input,tgt,tgt_mask = tgt_mask)

#             # dec_pred = (pred[:, 1:, 0:2] * std.to(device) + mean.to(device)).detach().cpu().numpy().cumsum(1) + batch[
#             #                                                                                                         'src'][                                                                                                         :, -1:,
#             #                                                                                             0:2].cpu().numpy()

#             #y = (batch['trg'][:, :, 2:4].to(device)).contiguous().view(-1, 2)
#             if loss_mode=='pair_wise':
#                 # remeber decoder_out is not being optimized but sigmas and mus
#                 loss = F.pairwise_distance(decoder_out[:, :,0:2].contiguous().view(-1, 2),((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(decoder_out[:,:,2]))
               
#             elif(loss_mode=='msq'):
#                 # remeber decoder_out is not being optimized but sigmas and mus
#                 pred = decoder_out[:, :,0:2].contiguous().view(-1, 2)
#                 y = ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device) 
#                 loss = nn.MSELoss(pred,y)
#             elif(loss_mode=='mdn'):
#                 if normalized:
#                     y = (batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)
#                 else:
#                     y = batch['trg'][:, :, 2:4].to(device)

#                 batch_pred, batch_gt = sampler(pi, sigma_x,sigma_y, mu_x , mu_y,batch['trg'][:, :, 0:2],mixtures=mixtures)

#                 #batch_pred, batch_gt = sampler(pi, sigma_x,sigma_y, mu_x , mu_y,batch['trg'][:, :, 0:2],mixtures=mixtures)
#                 # batch_gt = batch_gt.detach().cpu().numpy()
#                 #batch_pred = (batch_pred[:, :,:] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + batch[ 'src'][ :, -1:,0:2].cpu().numpy()
#                 # batch_pred = torch.tensor(batch_pred)
#                 #print("batch_pred:",type(batch_pred),batch_pred.shape,type(y),y.shape)
#                 # print("batch_gt:",type(batch_gt),(batch_gt.reshape(-1,2)).shape,(batch_pred.reshape(-1,2)).shape)
#                 loss_mdn = mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,y,mixtures,device)
#                 msq = Mean_squared_distance(y.contiguous().view(-1, 2).to(device),batch_pred.to(device).contiguous().view(-1, 2))
#                 #msq = F.pairwise_distance(torch.tensor(batch_pred).contiguous().view(-1, 2).to(device),((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() #+ torch.mean(torch.abs(torch.tensor(batch_pred[:,:,2]))) 
#                 #nn.MSELoss(torch.tensor(batch_pred.reshape(-1,2)),torch.tensor(batch_gt.reshape(-1,2)))
#                 #train_loss = loss_mdn
#                 #print("shape: ",batch_pred.contiguous().view(-1, 2).shape)
#                 #msq = F.pairwise_distance(batch_pred.contiguous().view(-1, 2).to(device),y.contiguous().view(-1, 2).to(device)).mean()
#                 # print("mdn_weight: ",model.mdn_weight)*
#                 train_loss =  model.mdn_weight * loss_mdn + (1- model.mdn_weight)*msq
#                 #print("loss_mdn: ",loss_mdn,"   loss_mdn: ",msq ," after: ",(model.mdn_weight * loss_mdn),(loss_mdn + (1- model.mdn_weight)*msq))
                
    
            
#             #print("shape: ",batch_pred.contiguous().view(-1, 2).shape)
#             train_loss.backward()
#             # Update the learning rate schedule
#             optim.step_and_update_lr()

#             epoch_loss += train_loss.item()


#             # gts_train.append(batch['trg'][:, :, 0:2]) 
#             # preds_train.append(dec_pred)

        
#         avg_loss_epoch = epoch_loss / num_batches
#         loss_list.append(avg_loss_epoch)
        
#         lr = optim._optimizer.param_groups[0]['lr']

#         print(f"Train loss:{avg_loss_epoch:.4f} mdn weight: {model.mdn_weight:.4f}") #mdn weighte: {model.mdn_weight}


#         ## EVALUATE in validation

#         with torch.no_grad():
#             model.eval()
#             gts_ev,preds_ev,src_ev = [], [] ,[]

#             for id_e, val_batch in enumerate(val_dl):
#                 #load_val.set_description(f"Epoch: {epoch+1} / {epochs}")
#                 src_ev.append(val_batch['src'])
#                 gts_ev.append(val_batch['trg'][:, :, 0:2])

#                 if(normalized):
#                     inp_val=(val_batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
#                     target_val=(val_batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
#                     y_val = (val_batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)

#                     # input_valc = torch.sqrt(torch.square(val_batch['src'][:,1:,2].to(device)) + torch.square(val_batch['src'][:,1:,3].to(device))).unsqueeze(-1)

#                     # input_val = torch.cat((inp_val,input_valc),-1)
#                 else:
#                     inp_val = val_batch['src'][:,1:,2:4].to(device)
#                     target_val = val_batch['trg'][:,:-1,2:4].to(device)
#                     y_val = val_batch['trg'][:, :, 2:4].to(device)

#                 if (add_features):                

#                     input_valc = torch.sqrt(torch.square(val_batch['src'][:,1:,2].to(device)) + torch.square(val_batch['src'][:,1:,3].to(device))).unsqueeze(-1)
#                     input_val = torch.cat((inp_val,input_valc),-1)
#                     # input_vald = (val_batch['src'][:,1:,3].to(device) / val_batch['src'][:,1:,2].to(device)).unsqueeze(-1)
#                     # input_val = torch.cat((input_val,input_vald),-1)
#                 else:
#                     input_val = inp_val
#                     target_val = target_val

#                 # target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
#                 # target=torch.cat((target,target_c),-1)
#                 # start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp_val.shape[0],1,1).to(device)
#                 # end_of_seq = torch.Tensor([0, 0, 0]).unsqueeze(0).unsqueeze(1).repeat(target_val.shape[0],11,1).to(device)
#                 #tgt_val = torch.cat((start_of_seq, end_of_seq), 1)


#                 tgt_val = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(target_val.shape[0],12,1).to(device)

               
                
#                 tgt_val_mask = generate_square_mask(dim_trg = decoder_seq_len ,dim_src = encoder_seq_len, mask_type="tgt").to(device)

#                 pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = model(input_val,tgt_val,tgt_mask = tgt_val_mask)

#                 #params = [pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out]

       
#                 loss_val_mdn = mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,y_val,mixtures,device)
#                 #msq = F.pairwise_distance(torch.tensor(batch_pred).contiguous().view(-1, 2).to(device),((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() #+ torch.mean(torch.abs(torch.tensor(batch_pred[:,:,2]))) 
#                 #nn.MSELoss(torch.tensor(batch_pred.reshape(-1,2)),torch.tensor(batch_gt.reshape(-1,2)))
                
#                 # print("mdn_weight: ",model.mdn_weight)
#                 batch_pred_val, batch_gt_val = sampler(pi, sigma_x,sigma_y, mu_x , mu_y,val_batch['trg'][:, :, 0:2],mixtures=mixtures)
#                 msq_val = Mean_squared_distance(y_val.contiguous().view(-1, 2).to(device),batch_pred_val.contiguous().view(-1, 2).to(device))
#                 #msq_val = F.pairwise_distance(batch_pred_val.contiguous().view(-1, 2).to(device),y_val.contiguous().view(-1, 2).to(device)).mean()
#                 loss_val =  model.mdn_weight * loss_val_mdn + (1- model.mdn_weight)*msq_val
#                 epoch_val_loss += loss_val.item()



#                 batch_gt_val = batch_gt_val.detach().cpu().numpy()
            
#                 if (post_process and normalized):
#                 #print("TRUE")
#                     batch_pred_val = (batch_pred_val[:, :,:] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + val_batch[
#                                                                                                                 'src'][
#                                                                                                             :, -1:,
#                                                                                                             0:2].cpu().numpy()
#                 else:
#                     batch_pred_val = batch_pred_val.detach().cpu().numpy()
#                     batch_pred_val = batch_pred_val[:, :,:].cumsum(1) + val_batch['src'][:, -1:,0:2].cpu().numpy()
#                 # calcualte error:
 
#                 mad, fad, errs = distance_metrics(batch_gt_val, batch_pred_val)
#                 # print('Eval/MAD', mad)
#                 # print('Eval/FAD', fad)
#                 val_epoch_mad.append(mad)
#                 val_epoch_fad.append(fad)

#                 #tgt = start_of_seq

#                 # for i in range(decoder_seq_len):
#                 #     tgt_mask = generate_square_mask(dim_trg = tgt.shape[1] ,dim_src = tgt.shape[1], mask_type="tgt").to(device)
#                 #     #tgt_mask = subsequent_mask(tgt.shape[1]).to(device)
#                 #     pred = model(inp,tgt,tgt_mask = tgt_mask)
#                 #     tgt=torch.cat((tgt,pred[:,-1:,:]),1)
#                 #preds_eval=(tgt[:,1:,0:2]*std.to(device)+mean.to(device)).detach().cpu().numpy().cumsum(1)+batch['src'][:,-1:,0:2].detach().cpu().numpy()
#                 #preds_ev.append(preds_eval)
 
#             # gts_val = np.concatenate(gts_ev, 0)  
#             # preds_val = np.concatenate(preds_ev, 0)

#             # mad, fad, errs = distance_metrics(gts_val, preds_val)
#             # all_mad.append(mad)
#             # all_fad.append(fad)
#             # print('Eval/MAD', mad)
#             # print('Eval/FAD', fad)
        
#         avg_loss_epoch_val = epoch_val_loss / num_batches_val
#         sum(val_epoch_mad)
#         val_batch_mad = (sum(val_epoch_mad)/num_batches_val)
#         val_batch_fad = (sum(val_epoch_fad)/num_batches_val)
#         val_mad.append(val_batch_mad)
#         val_fad.append(val_batch_fad)
#         loss_eval.append(avg_loss_epoch_val)

#         # save and stop model
#         early_stop(model,avg_loss_epoch, avg_loss_epoch_val,epoch+1,val_batch_mad,val_batch_fad)
#         print(f"Eval loss: {avg_loss_epoch_val:.4f}")
#         print(f"Learning Rate: {lr:.5f}")
        
#         mad_test , fad_test,_,_,mdn_results,avg_mad,avg_fad,mdn_res= attenion_mdn_test(test_dl, model,device,add_features = add_features,mixtures=mixtures,enc_seq = 8,dec_seq=12, mode='feed',loss_mode ='mdn',mean=mean,std=std)
#         test_mad.append(avg_mad)
#         test_fad.append(avg_fad)

#         # if (early_stop.stop):
#         #     print("Early stopping activated!")
#         #     break

        

#     print(f"Epoch {epoch+1} Train loss: {avg_loss_epoch}")
#     print(f"Epoch {epoch+1} Eval loss: {avg_loss_epoch_val}")
#     return loss_list, loss_eval, test_mad , test_fad,val_mad,val_fad#all_mad,all_fad,loss_list




def train_prob_attn_mdn(train_dl,val_dl,test_dl, model, optim,add_features = False,mixtures = 3,device=device_train, epochs=20,enc_seq = 8,dec_seq=12,post_process= True,normalized ='True', mode='feed',loss_mode ='mdn',mean=0,std=0):
    early_stop = early_save_stop()
    encoder_seq_len = enc_seq
    decoder_seq_len = dec_seq  
    num_batches = len(train_dl)
    num_batches_val = len(val_dl)

    loss_list,loss_eval,all_mad,all_fad = [],[],[],[]
    val_mad,val_fad,test_mad,test_fad = [],[],[],[]
    print('Training Setting:... ')
    print(f"Train batch size {num_batches}\nloss: {loss_mode}\nData Normalized: {normalized}")
    for epoch in range (epochs):
        gts_train, preds_train = [] , []
        epoch_loss,epoch_val_loss = 0 , 0
        val_epoch_mad,val_epoch_fad = [],[]
        load_train = tqdm(train_dl)
        # load_val = tqdm(val_dl)

        model.train()
        for id_b,batch in enumerate(load_train):

            load_train.set_description(f"Epoch: {epoch+1} / {epochs}")

            #src_mask = generate_square_mask(dim_trg = encoder_seq_len ,dim_src = decoder_seq_len,mask_type="memory").to(device)

            if (normalized):
                
                inp=(batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
                target=(batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)


            else:
                inp=(batch['src'][:,1:,2:4].to(device).to(device))
                target=(batch['trg'][:,:-1,2:4].to(device).to(device))
            
            if (add_features):
                
                input_c = torch.sqrt(torch.square(batch['src'][:,1:,2].to(device)) + torch.square(batch['src'][:,1:,3].to(device))).unsqueeze(-1)
                #input_d = (batch['src'][:,1:,3].to(device) / batch['src'][:,1:,2].to(device)).unsqueeze(-1)
                input = torch.cat((inp,input_c),-1)
                # input = torch.cat((input_temp,input_d),-1)
                # print("input.shape: ", input.shape)
            else:
                input = inp
                target = target


            target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
            target=torch.cat((target,target_c),-1)
            # start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)
            # end_of_seq = torch.Tensor([0, 0, 0]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],11,1).to(device)
            # tgt = torch.cat((start_of_seq, end_of_seq), 1)

            #tgt = torch.cat((start_of_seq, target), 1)
            

            # print("This is target shape: ",tgt.shape,'\ntgt: ',tgt[0])

            #src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            #trg_att=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)
            tgt = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],12,1).to(device)

            tgt_mask = generate_square_mask(dim_trg = decoder_seq_len ,dim_src = encoder_seq_len, mask_type="tgt").to(device)
            #tgt_mask = subsequent_mask(decoder_seq_len).to(device)

            optim.zero_grad()
            pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = model(input,tgt,tgt_mask = tgt_mask)

            # dec_pred = (pred[:, 1:, 0:2] * std.to(device) + mean.to(device)).detach().cpu().numpy().cumsum(1) + batch[
            #                                                                                                         'src'][                                                                                                         :, -1:,
            #                                                                                             0:2].cpu().numpy()

            #y = (batch['trg'][:, :, 2:4].to(device)).contiguous().view(-1, 2)
            if loss_mode=='pair_wise':
                # remeber decoder_out is not being optimized but sigmas and mus
                loss = F.pairwise_distance(decoder_out[:, :,0:2].contiguous().view(-1, 2),((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(decoder_out[:,:,2]))
               
            elif(loss_mode=='msq'):
                # remeber decoder_out is not being optimized but sigmas and mus
                pred = decoder_out[:, :,0:2].contiguous().view(-1, 2)
                y = ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device) 
                loss = nn.MSELoss(pred,y)
            elif(loss_mode=='mdn'):
                if normalized:
                    y = (batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)
                else:
                    y = batch['trg'][:, :, 2:4].to(device)

                batch_pred, batch_gt = sampler(pi, sigma_x,sigma_y, mu_x , mu_y,batch['trg'][:, :, 0:2],mixtures=mixtures)

                #batch_pred, batch_gt = sampler(pi, sigma_x,sigma_y, mu_x , mu_y,batch['trg'][:, :, 0:2],mixtures=mixtures)
                # batch_gt = batch_gt.detach().cpu().numpy()
                #batch_pred = (batch_pred[:, :,:] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + batch[ 'src'][ :, -1:,0:2].cpu().numpy()
                # batch_pred = torch.tensor(batch_pred)
                #print("batch_pred:",type(batch_pred),batch_pred.shape,type(y),y.shape)
                # print("batch_gt:",type(batch_gt),(batch_gt.reshape(-1,2)).shape,(batch_pred.reshape(-1,2)).shape)
                loss_mdn = mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,y,mixtures,device)
                msq = Mean_squared_distance(y.contiguous().view(-1, 2).to(device),batch_pred.to(device).contiguous().view(-1, 2))
                #msq = F.pairwise_distance(torch.tensor(batch_pred).contiguous().view(-1, 2).to(device),((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() #+ torch.mean(torch.abs(torch.tensor(batch_pred[:,:,2]))) 
                #nn.MSELoss(torch.tensor(batch_pred.reshape(-1,2)),torch.tensor(batch_gt.reshape(-1,2)))
                #train_loss = loss_mdn
                #print("shape: ",batch_pred.contiguous().view(-1, 2).shape)
                #msq = F.pairwise_distance(batch_pred.contiguous().view(-1, 2).to(device),y.contiguous().view(-1, 2).to(device)).mean()
                # print("mdn_weight: ",model.mdn_weight)*
                train_loss =  model.mdn_weight * loss_mdn + (1- model.mdn_weight)*msq
                #print("loss_mdn: ",loss_mdn,"   loss_mdn: ",msq ," after: ",(model.mdn_weight * loss_mdn),(loss_mdn + (1- model.mdn_weight)*msq))
                
    
            
            #print("shape: ",batch_pred.contiguous().view(-1, 2).shape)
            train_loss.backward()
            # Update the learning rate schedule
            optim.step_and_update_lr()

            epoch_loss += train_loss.item()


            # gts_train.append(batch['trg'][:, :, 0:2]) 
            # preds_train.append(dec_pred)

        
        avg_loss_epoch = epoch_loss / num_batches
        loss_list.append(avg_loss_epoch)
        
        lr = optim._optimizer.param_groups[0]['lr']

        print(f"Train loss:{avg_loss_epoch:.4f} mdn weight: {model.mdn_weight:.4f}") #mdn weighte: {model.mdn_weight}


        ## EVALUATE in validation

        with torch.no_grad():
            model.eval()
            gts_ev,preds_ev,src_ev = [], [] ,[]

            for id_e, val_batch in enumerate(val_dl):
                #load_val.set_description(f"Epoch: {epoch+1} / {epochs}")
                src_ev.append(val_batch['src'])
                gts_ev.append(val_batch['trg'][:, :, 0:2])

                if(normalized):
                    inp_val=(val_batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
                    target_val=(val_batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
                    y_val = (val_batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)

                    # input_valc = torch.sqrt(torch.square(val_batch['src'][:,1:,2].to(device)) + torch.square(val_batch['src'][:,1:,3].to(device))).unsqueeze(-1)

                    # input_val = torch.cat((inp_val,input_valc),-1)
                else:
                    inp_val = val_batch['src'][:,1:,2:4].to(device)
                    target_val = val_batch['trg'][:,:-1,2:4].to(device)
                    y_val = val_batch['trg'][:, :, 2:4].to(device)

                if (add_features):                

                    input_valc = torch.sqrt(torch.square(val_batch['src'][:,1:,2].to(device)) + torch.square(val_batch['src'][:,1:,3].to(device))).unsqueeze(-1)
                    input_val = torch.cat((inp_val,input_valc),-1)
                    # input_vald = (val_batch['src'][:,1:,3].to(device) / val_batch['src'][:,1:,2].to(device)).unsqueeze(-1)
                    # input_val = torch.cat((input_val,input_vald),-1)
                else:
                    input_val = inp_val
                    target_val = target_val

                # target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
                # target=torch.cat((target,target_c),-1)
                # start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp_val.shape[0],1,1).to(device)
                # end_of_seq = torch.Tensor([0, 0, 0]).unsqueeze(0).unsqueeze(1).repeat(target_val.shape[0],11,1).to(device)
                #tgt_val = torch.cat((start_of_seq, end_of_seq), 1)


                tgt_val = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(target_val.shape[0],12,1).to(device)

               
                
                tgt_val_mask = generate_square_mask(dim_trg = decoder_seq_len ,dim_src = encoder_seq_len, mask_type="tgt").to(device)

                pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = model(input_val,tgt_val,tgt_mask = tgt_val_mask)

                #params = [pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out]

       
                loss_val_mdn = mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,y_val,mixtures,device)

                

             
                
                #msq = F.pairwise_distance(torch.tensor(batch_pred).contiguous().view(-1, 2).to(device),((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() #+ torch.mean(torch.abs(torch.tensor(batch_pred[:,:,2]))) 
                #nn.MSELoss(torch.tensor(batch_pred.reshape(-1,2)),torch.tensor(batch_gt.reshape(-1,2)))
                
                # print("mdn_weight: ",model.mdn_weight)
                


                batch_pred_val, batch_gt_val = sampler(pi, sigma_x,sigma_y, mu_x , mu_y,val_batch['trg'][:, :, 0:2],mixtures=mixtures)
                msq_val = Mean_squared_distance(y_val.contiguous().view(-1, 2).to(device),batch_pred_val.contiguous().view(-1, 2).to(device))
                #msq_val = F.pairwise_distance(batch_pred_val.contiguous().view(-1, 2).to(device),y_val.contiguous().view(-1, 2).to(device)).mean()
                loss_val =  model.mdn_weight * loss_val_mdn + (1- model.mdn_weight)*msq_val
                epoch_val_loss += loss_val.item()



                batch_gt_val = batch_gt_val.detach().cpu().numpy()
            
                if (post_process and normalized):
                #print("TRUE")
                    batch_pred_val = (batch_pred_val[:, :,:] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + val_batch[
                                                                                                                'src'][
                                                                                                            :, -1:,
                                                                                                            0:2].cpu().numpy()
                else:
                    batch_pred_val = batch_pred_val.detach().cpu().numpy()
                    batch_pred_val = batch_pred_val[:, :,:].cumsum(1) + val_batch['src'][:, -1:,0:2].cpu().numpy()
                # calcualte error:
 
                mad, fad, errs = distance_metrics(batch_gt_val, batch_pred_val)
                # print('Eval/MAD', mad)
                # print('Eval/FAD', fad)
                val_epoch_mad.append(mad)
                val_epoch_fad.append(fad)

                #tgt = start_of_seq

                # for i in range(decoder_seq_len):
                #     tgt_mask = generate_square_mask(dim_trg = tgt.shape[1] ,dim_src = tgt.shape[1], mask_type="tgt").to(device)
                #     #tgt_mask = subsequent_mask(tgt.shape[1]).to(device)
                #     pred = model(inp,tgt,tgt_mask = tgt_mask)
                #     tgt=torch.cat((tgt,pred[:,-1:,:]),1)
                #preds_eval=(tgt[:,1:,0:2]*std.to(device)+mean.to(device)).detach().cpu().numpy().cumsum(1)+batch['src'][:,-1:,0:2].detach().cpu().numpy()
                #preds_ev.append(preds_eval)
 
            # gts_val = np.concatenate(gts_ev, 0)  
            # preds_val = np.concatenate(preds_ev, 0)

            # mad, fad, errs = distance_metrics(gts_val, preds_val)
            # all_mad.append(mad)
            # all_fad.append(fad)
            # print('Eval/MAD', mad)
            # print('Eval/FAD', fad)
        
        avg_loss_epoch_val = epoch_val_loss / num_batches_val
        sum(val_epoch_mad)
        val_batch_mad = (sum(val_epoch_mad)/num_batches_val)
        val_batch_fad = (sum(val_epoch_fad)/num_batches_val)
        val_mad.append(val_batch_mad)
        val_fad.append(val_batch_fad)
        loss_eval.append(avg_loss_epoch_val)

        # save and stop model
        early_stop(model,avg_loss_epoch, avg_loss_epoch_val,epoch+1,val_batch_mad,val_batch_fad)
        print(f"Eval loss: {avg_loss_epoch_val:.4f}")
        print(f"Learning Rate: {lr:.5f}")
        
        mad_test , fad_test,_,_,mdn_results,avg_mad,avg_fad,mdn_res= attenion_mdn_test(test_dl, model,device,add_features = add_features,mixtures=mixtures,enc_seq = 8,dec_seq=12, mode='feed',loss_mode ='mdn',mean=mean,std=std)
        test_mad.append(avg_mad)
        test_fad.append(avg_fad)

        # if (early_stop.stop):
        #     print("Early stopping activated!")
        #     break

        

    print(f"Epoch {epoch+1} Train loss: {avg_loss_epoch}")
    print(f"Epoch {epoch+1} Eval loss: {avg_loss_epoch_val}")
    return loss_list, loss_eval, test_mad , test_fad,val_mad,val_fad#all_mad,all_fad,loss_list
    