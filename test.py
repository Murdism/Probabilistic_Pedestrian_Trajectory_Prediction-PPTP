
import torch 
import torch.nn as nn 
import numpy as np
from config import CFG
from loss import mdn_loss_fn,Mean_squared_distance
from utils import generate_square_mask,subsequent_mask,Err,traj_err
from tqdm import tqdm
import torch.nn.functional as F  
from baselineUtils import distance_metrics
from mdn import sampler,sampler_all_mean,sample_mean
from trajectory_candidates import run_cluster





device_test = CFG.device
batch_size = CFG.batch_size

def test_mdn(test_dl, model,device = device_test,sample='True',add_features = False,mixtures = 3,enc_seq = 8,dec_seq=12, mode='feed',normalized = True,loss_mode ='mdn',mean=0,std=0,post_process=True, db=False):
    # sample='True' means will sample false means return means instead
    ''' mode:  refers to the loss mode to be used :
            mdn: mdn loss
        db : use of dbscn clustering -> Default false'''
    
    all_time = []
    all_total_time = []
    all_average = []
    total_time = 0
    index_batch = 1 
    with torch.no_grad():
        model.eval()
        gts_ev,batch_preds,batch_gts,src_ev = [], [] , [],[]
        mdn_results = []
        sum_mad , sum_fad = [] , []
        acc_mad , acc_fad = [] , []
        mean_error = 0
        loss_eval = []
        candidate_trajs,best_candiates,src_trajs,candidate_weights = [],[],[],[]
        sum_mad_best,sum_fad_best = [],[]
        epoch_val_loss = 0
        epoch_val_loss_combined = 0
        num_batches_val = len(test_dl)
        load_test = tqdm(test_dl)

        for id_e, val_batch in enumerate(load_test):
            load_test.set_description(f"Batch: {id_e+1} / {num_batches_val}")
            src_ev.append(val_batch['src'])
            gts_ev.append(val_batch['trg'][:, :, 0:2])
            batch_gt = val_batch['trg'][:, :, 0:2].to(device).cpu().numpy()

            if (normalized):
                inp_val=(val_batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
                target_val=(val_batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
                y_val = (val_batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)

                input_valc = torch.sqrt(torch.square(val_batch['src'][:,1:,2].to(device)) + torch.square(val_batch['src'][:,1:,3].to(device))).unsqueeze(-1)
                input_val = torch.cat((inp_val,input_valc),-1)

            else :
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

            tgt_val = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(target_val.shape[0],12,1).to(device)


            

            tgt_val_mask = generate_square_mask(dim_trg = dec_seq ,dim_src = enc_seq, mask_type="tgt").to(device)
            
            # s = time.time()

            pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = model(input_val,tgt_val,tgt_mask = tgt_val_mask)
            # curr_time = (time.time()-s )*1000  
            # total_time = curr_time + total_time  
            # average_time_batch =  total_time / index_batch   
            # index_batch = index_batch + 1  
            # print("--------------------------------Index-------------------------------- : " , index_batch-1)
            # print("--------------------------------Time Taken-------------------------------- : " , curr_time)
            # print("--------------------------------Total Time Taken-------------------------------- : " , total_time)
            # print("--------------------------------Average Time Taken-------------------------------- : " , average_time_batch)


            
            mus = torch.cat((mu_x.unsqueeze(-1),mu_y.unsqueeze(-1)),-1)
            sigmas = torch.cat((sigma_x.unsqueeze(-1),sigma_y.unsqueeze(-1)),-1)

            src_value = val_batch[ 'src'][:, -1:,0:2].detach().cpu().numpy()
            src_value = src_value[:,np.newaxis,:,:]
            cluster_mus = (mus[:, :,:] * std.to(device) + mean.to(device)).detach().cpu().numpy().cumsum(1) + src_value

            cluster_real = val_batch['trg'][:, :, 0:2]
            cluster_src = val_batch['src'][:,:,0:2]


            #batch_trajs,best_trajs = run_cluster(cluster_mus,pi,cluster_real,cluster_src,dbscan=db)
            batch_trajs,batch_weights,best_trajs,best_weights = run_cluster(cluster_mus,pi,cluster_real,cluster_src,dbscan=db)
            ades,fdes = 0,0
            counter = 0
            for trajs,wgts,cluster in zip(batch_trajs,batch_weights,cluster_real):
                # for traj,wgt in zip(trajs,wgts):
                traj_ade = 0
                traj_fde = 0
                for traj,wgt in zip(trajs,wgts):
                    ade,fde = traj_err(cluster,traj)
                    traj_ade+=wgt*ade
                    traj_fde+=wgt*fde
                ades+=traj_ade
                fdes+=traj_fde
                counter+=1
            # print("accumlated: ",ades/counter,fdes/counter)
            acc_mad.append(ades/counter)
            acc_fad.append(fdes/counter)

            
            candidate_trajs.append(batch_trajs)
            candidate_weights.append(batch_weights)
            best_candiates.append(best_trajs)
            src_trajs.append(cluster_src)
            # best_trajs = best_trajs[0]
            # print(len(best_trajs),len(best_trajs[0]),len(best_trajs[0][0]))
            # print("cluster_real: ",cluster_real.shape)
            # print("cluster_real: ",batch_gt.shape)


            batch_pred = sample_mean(pi,sigmas,mus)


            #calculate_loss
            loss_val = mdn_loss_fn(pi, sigma_x,sigma_y, mu_x , mu_y,y_val,mixtures)
            msq = Mean_squared_distance(y_val.contiguous().view(-1, 2).to(device),batch_pred.to(device).contiguous().view(-1, 2))
            combined_loss = 0.5 * loss_val + (1- 0.5)*msq

            epoch_val_loss += loss_val.item()
            epoch_val_loss_combined +=combined_loss.item()


            # print("----------Best_trajs-----------: ",len(best_trajs),len(best_trajs[0]),len(best_trajs[0][0]))
            # best_trajs = torch.tensor(best_trajs).to(device)
            if (normalized):
                #print("-----------TRUE---------batch_pred: ",type(batch_pred),len(batch_pred),len(batch_pred[0]),len(batch_pred[0][0]))
                batch_pred = (batch_pred[:, :,:] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + val_batch[
                                                                                                                'src'][
                                                                                                            :, -1:,
                                                                                                            0:2].cpu().numpy()
                
            else:
                batch_pred = batch_pred.detach().cpu().numpy()
                batch_pred = batch_pred[:, :,:].cumsum(1) + val_batch['src'][:, -1:,0:2].cpu().numpy()
            # calcualte error:
            #print("batch_gt: ",batch_gt[0],"\nbatch_pred: ",batch_pred[0])
            # mad, fad, errs = distance_metrics(batch_gt, batch_pred)
            # if(not (batch_pred[0]==best_trajs[0]).any()):
            #     print("batch_pred",batch_pred[0])
            #     print("best_trajs",best_trajs[0])
            # print("----------------------------")
            # print("batch_pred",batch_pred[0])
            # print("best_trajs",best_trajs[0])     

            # batch_pred = (batch_pred[:, :,:] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + val_batch[
            #                                                                                                     'src'][
            #                                                                                                 :, -1:,
            #                                                                                                 0:2].cpu().numpy()

            # print("batch_pred==best_trajs: ",batch_pred==best_trajs)
           
            mad, fad = Err(batch_gt.tolist(), batch_pred.tolist())
            # print('Eval/MAD', mad)
            # print('Eval/FAD', fad)
            #cluster_real
            mad_best, fad_best = Err(cluster_real.tolist(), best_trajs)

            sum_mad.append(mad)
            sum_fad.append(fad)
            sum_mad_best.append(mad_best)
            sum_fad_best.append(fad_best)
            # print('Eval/mad_best', mad_best)
            # print('Eval/fad_best', fad_best)
            batch_preds.append(batch_pred)
            batch_gts.append(batch_gt)

    avg_loss_epoch_val = epoch_val_loss / num_batches_val
    avg_loss_epoch_val_combined = epoch_val_loss_combined / num_batches_val
    avg_mad = sum(sum_mad)/num_batches_val
    avg_fad = sum(sum_fad)/num_batches_val
    avg_mad_best = sum(sum_mad_best)/num_batches_val
    avg_fad_best = sum(sum_fad_best)/num_batches_val

    avg_acc_mad = sum(acc_mad)/num_batches_val
    avg_acc_fad = sum(acc_fad)/num_batches_val
    loss_eval.append(avg_loss_epoch_val)
    print(f"Test loss: {avg_loss_epoch_val}") 
    print(f"Test loss with msq: { avg_loss_epoch_val_combined}") 
    print(f"Test avg_mad: {avg_mad}")
    print(f"Test avg_fad: {avg_fad}")
    print(f"Test avg_mad_best: {avg_mad_best}")
    print(f"Test avg_fad_best: {avg_fad_best}")
    print(f"Test avg_acc_mad: {avg_acc_mad}")
    print(f"Test avg_acc_fad: {avg_acc_fad}")
    

    return batch_preds,batch_gts,avg_mad,avg_fad,candidate_trajs,candidate_weights,best_candiates,src_trajs



def inference_realtime(test_dl, model,device = device_test,sample='True',add_features = False,mixtures = 3,enc_seq = 8,dec_seq=12, mode='feed',normalized = True,loss_mode ='mdn',mean=0,std=0,post_process=True, db=False):
    # sample='True' means will sample false means return means instead
    ''' mode:  refers to the loss mode to be used :
            mdn: mdn loss
        db : use of dbscn clustering -> Default false'''
    
    with torch.no_grad():
        model.eval()
        gts_ev,batch_preds,batch_gts,src_ev = [], [] , [],[]
        candidate_trajs,best_candiates,src_trajs,candidate_weights = [],[],[],[]
        sum_mad_best,sum_fad_best = [],[]
        epoch_val_loss = 0
        epoch_val_loss_combined = 0
        num_batches_val = len(test_dl)
        #load_test = tqdm(test_dl)

        for id_e, val_batch in enumerate(test_dl):
            #load_test.set_description(f"Batch: {id_e+1} / {num_batches_val}")
            src_ev.append(val_batch)
            # gts_ev.append(val_batch['trg'][:, :, 0:2])
            # batch_gt = val_batch['trg'][:, :, 0:2].to(device).cpu().numpy()

            if (normalized):
                inp_val=(val_batch[:,1:,2:4].to(device)-mean.to(device))/std.to(device)
                # target_val=(val_batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
                # y_val = (val_batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)

                input_valc = torch.sqrt(torch.square(val_batch[:,1:,2].to(device)) + torch.square(val_batch[:,1:,3].to(device))).unsqueeze(-1)
                input_val = torch.cat((inp_val,input_valc),-1)

            else :
                inp_val = val_batch[:,1:,2:4].to(device)
                # target_val = val_batch['trg'][:,:-1,2:4].to(device)
                # y_val = val_batch['trg'][:, :, 2:4].to(device)


            if (add_features):                

                    input_valc = torch.sqrt(torch.square(val_batch[:,1:,2].to(device)) + torch.square(val_batch[:,1:,3].to(device))).unsqueeze(-1)
                    input_val = torch.cat((inp_val,input_valc),-1)
                    # input_vald = (val_batch['src'][:,1:,3].to(device) / val_batch['src'][:,1:,2].to(device)).unsqueeze(-1)
                    # input_val = torch.cat((input_val,input_vald),-1)
            else:
                    input_val = inp_val
                    target_val = target_val

            tgt_val = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(inp_val.shape[0],12,1).to(device)


            

            tgt_val_mask = generate_square_mask(dim_trg = dec_seq ,dim_src = enc_seq, mask_type="tgt").to(device)
            


            pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = model(input_val,tgt_val,tgt_mask = tgt_val_mask)

            mus = torch.cat((mu_x.unsqueeze(-1),mu_y.unsqueeze(-1)),-1)
            sigmas = torch.cat((sigma_x.unsqueeze(-1),sigma_y.unsqueeze(-1)),-1)

            src_value = val_batch[:, -1:,0:2].detach().cpu().numpy()
            src_value = src_value[:,np.newaxis,:,:]
            cluster_mus = (mus[:, :,:] * std.to(device) + mean.to(device)).detach().cpu().numpy().cumsum(1) + src_value

            # cluster_real = val_batch['trg'][:, :, 0:2]
            cluster_real = val_batch[:, :, 0:2]
            cluster_src = val_batch[:,:,0:2]


            batch_trajs,batch_weights,best_trajs,best_weights = run_cluster(cluster_mus,pi,cluster_real,cluster_src,dbscan=db)


            
            candidate_trajs.append(batch_trajs)
            candidate_weights.append(batch_weights)
            best_candiates.append(best_trajs)
            src_trajs.append(cluster_src)



            batch_pred = sample_mean(pi,sigmas,mus)
            
            if (normalized):
                #print("TRUE")
                batch_pred = (batch_pred[:, :,:] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + val_batch[
                                                                                                            :, -1:,
                                                                                                            0:2].cpu().numpy()
                
            else:
                batch_pred = batch_pred.detach().cpu().numpy()
                batch_pred = batch_pred[:, :,:].cumsum(1) + val_batch['src'][:, -1:,0:2].cpu().numpy()
            batch_preds.append(batch_pred)
    
    return batch_preds,candidate_trajs,candidate_weights,best_candiates,src_trajs


def inference(test_dl, model,device = device_test,sample='True',add_features = False,mixtures = 3,enc_seq = 8,dec_seq=12, mode='feed',normalized = True,loss_mode ='mdn',mean=0,std=0,post_process=True, db=False):
    # sample='True' means will sample false means return means instead
    ''' mode:  refers to the loss mode to be used :
            mdn: mdn loss
        db : use of dbscn clustering -> Default false'''
    
    with torch.no_grad():
        model.eval()
        gts_ev,batch_preds,batch_gts,src_ev = [], [] , [],[]
        candidate_trajs,best_candiates,src_trajs,candidate_weights = [],[],[],[]
        sum_mad_best,sum_fad_best = [],[]
        epoch_val_loss = 0
        epoch_val_loss_combined = 0
        num_batches_val = len(test_dl)
        load_test = tqdm(test_dl)

        for id_e, val_batch in enumerate(load_test):
            load_test.set_description(f"Batch: {id_e+1} / {num_batches_val}")
            src_ev.append(val_batch['src'])
            # gts_ev.append(val_batch['trg'][:, :, 0:2])
            # batch_gt = val_batch['trg'][:, :, 0:2].to(device).cpu().numpy()

            if (normalized):
                inp_val=(val_batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
                # target_val=(val_batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
                # y_val = (val_batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)

                input_valc = torch.sqrt(torch.square(val_batch['src'][:,1:,2].to(device)) + torch.square(val_batch['src'][:,1:,3].to(device))).unsqueeze(-1)
                input_val = torch.cat((inp_val,input_valc),-1)

            else :
                inp_val = val_batch['src'][:,1:,2:4].to(device)
                # target_val = val_batch['trg'][:,:-1,2:4].to(device)
                # y_val = val_batch['trg'][:, :, 2:4].to(device)


            if (add_features):                

                    input_valc = torch.sqrt(torch.square(val_batch['src'][:,1:,2].to(device)) + torch.square(val_batch['src'][:,1:,3].to(device))).unsqueeze(-1)
                    input_val = torch.cat((inp_val,input_valc),-1)
                    # input_vald = (val_batch['src'][:,1:,3].to(device) / val_batch['src'][:,1:,2].to(device)).unsqueeze(-1)
                    # input_val = torch.cat((input_val,input_vald),-1)
            else:
                    input_val = inp_val
                    target_val = target_val

            tgt_val = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(inp_val.shape[0],12,1).to(device)


            

            tgt_val_mask = generate_square_mask(dim_trg = dec_seq ,dim_src = enc_seq, mask_type="tgt").to(device)
            


            pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = model(input_val,tgt_val,tgt_mask = tgt_val_mask)

            mus = torch.cat((mu_x.unsqueeze(-1),mu_y.unsqueeze(-1)),-1)
            sigmas = torch.cat((sigma_x.unsqueeze(-1),sigma_y.unsqueeze(-1)),-1)

            src_value = val_batch[ 'src'][:, -1:,0:2].detach().cpu().numpy()
            src_value = src_value[:,np.newaxis,:,:]
            cluster_mus = (mus[:, :,:] * std.to(device) + mean.to(device)).detach().cpu().numpy().cumsum(1) + src_value

            # cluster_real = val_batch['trg'][:, :, 0:2]
            cluster_real = val_batch['src'][:, :, 0:2]
            cluster_src = val_batch['src'][:,:,0:2]


            batch_trajs,batch_weights,best_trajs,best_weights = run_cluster(cluster_mus,pi,cluster_real,cluster_src,dbscan=db)


            
            candidate_trajs.append(batch_trajs)
            candidate_weights.append(batch_weights)
            best_candiates.append(best_trajs)
            src_trajs.append(cluster_src)



            batch_pred = sample_mean(pi,sigmas,mus)
            
            if (normalized):
                #print("TRUE")
                batch_pred = (batch_pred[:, :,:] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + val_batch[
                                                                                                                'src'][
                                                                                                            :, -1:,
                                                                                                            0:2].cpu().numpy()
                
            else:
                batch_pred = batch_pred.detach().cpu().numpy()
                batch_pred = batch_pred[:, :,:].cumsum(1) + val_batch['src'][:, -1:,0:2].cpu().numpy()
            batch_preds.append(batch_pred)
    
    return batch_preds,candidate_trajs,candidate_weights,best_candiates,src_trajs
