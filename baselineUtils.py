# Source: https://github.com/FGiuliari/Trajectory-Transformer/blob/master/baselineUtils.py

from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
import random
import scipy.spatial
import scipy.io
from subs import subsequent_mask
from config import Args,CFG

def create_dataset(dataset_folder,dataset_name,val_size,gt,horizon,delim="\t",train=True,eval=False,verbose=True):

        if train==True:
            datasets_list = os.listdir(os.path.join(dataset_folder,dataset_name, "train"))
            full_dt_folder=os.path.join(dataset_folder,dataset_name, "train")
        if train==False and eval==False:
            datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, "val"))
            full_dt_folder = os.path.join(dataset_folder, dataset_name, "val")
        if train==False and eval==True:
            datasets_list = os.listdir(os.path.join(dataset_folder, dataset_name, "test"))
            full_dt_folder = os.path.join(dataset_folder, dataset_name, "test")


        datasets_list=datasets_list
        data={}
        data_src=[]
        data_trg=[]
        data_seq_start=[]
        data_frames=[]
        data_dt=[]
        data_peds=[]

        val_src = []
        val_trg = []
        val_seq_start = []
        val_frames = []
        val_dt = []
        val_peds=[]

        if verbose:
            print("start loading dataset")
            print("validation set size -> %i"%(val_size))


        for i_dt, dt in enumerate(datasets_list):
            if verbose:
                print("%03i / %03i - loading %s"%(i_dt+1,len(datasets_list),dt))
            raw_data = pd.read_csv(os.path.join(full_dt_folder, dt), delimiter=delim,
                                            names=["frame", "ped", "x", "y"],usecols=[0,1,2,3],na_values="?")

            raw_data.sort_values(by=['frame','ped'], inplace=True)

            inp,out,info=get_strided_data_clust(raw_data,gt,horizon,1)

            dt_frames=info['frames']
            dt_seq_start=info['seq_start']
            dt_dataset=np.array([i_dt]).repeat(inp.shape[0])
            dt_peds=info['peds']



            if val_size>0 and inp.shape[0]>val_size*2.5:
                if verbose:
                    print("created validation from %s" % (dt))
                k = random.sample(np.arange(inp.shape[0]).tolist(), val_size)
                val_src.append(inp[k, :, :])
                val_trg.append(out[k, :, :])
                val_seq_start.append(dt_seq_start[k, :, :])
                val_frames.append(dt_frames[k, :])
                val_dt.append(dt_dataset[k])
                val_peds.append(dt_peds[k])
                inp = np.delete(inp, k, 0)
                out = np.delete(out, k, 0)
                dt_frames = np.delete(dt_frames, k, 0)
                dt_seq_start = np.delete(dt_seq_start, k, 0)
                dt_dataset = np.delete(dt_dataset, k, 0)
                dt_peds = np.delete(dt_peds,k,0)
            elif val_size>0:
                if verbose:
                    print("could not create validation from %s, size -> %i" % (dt,inp.shape[0]))

            data_src.append(inp)
            data_trg.append(out)
            data_seq_start.append(dt_seq_start)
            data_frames.append(dt_frames)
            data_dt.append(dt_dataset)
            data_peds.append(dt_peds)





        data['src'] = np.concatenate(data_src, 0)
        data['trg'] = np.concatenate(data_trg, 0)
        data['seq_start'] = np.concatenate(data_seq_start, 0)
        data['frames'] = np.concatenate(data_frames, 0)
        data['dataset'] = np.concatenate(data_dt, 0)
        data['peds'] = np.concatenate(data_peds, 0)
        data['dataset_name'] = datasets_list

        mean= data['src'].mean((0,1))
        std= data['src'].std((0,1))

        if val_size>0:
            data_val={}
            data_val['src']=np.concatenate(val_src,0)
            data_val['trg'] = np.concatenate(val_trg, 0)
            data_val['seq_start'] = np.concatenate(val_seq_start, 0)
            data_val['frames'] = np.concatenate(val_frames, 0)
            data_val['dataset'] = np.concatenate(val_dt, 0)
            data_val['peds'] = np.concatenate(val_peds, 0)

            return IndividualTfDataset(data, "train", mean, std), IndividualTfDataset(data_val, "validation", mean, std)

        return IndividualTfDataset(data, "train", mean, std), None




        return IndividualTfDataset(data,"train",mean,std), IndividualTfDataset(data_val,"validation",mean,std)



class IndividualTfDataset(Dataset):
    def __init__(self,data,name,mean,std):
        super(IndividualTfDataset,self).__init__()

        self.data=data
        self.name=name

        self.mean= mean
        self.std = std

    def __len__(self):
        return self.data['src'].shape[0]


    def __getitem__(self,index):
        return {'src':torch.Tensor(self.data['src'][index]),
                'trg':torch.Tensor(self.data['trg'][index]),
                'frames':self.data['frames'][index],
                'seq_start':self.data['seq_start'][index],
                'dataset':self.data['dataset'][index],
                'peds': self.data['peds'][index],
                }







def create_folders(baseFolder,datasetName):
    try:
        os.mkdir(baseFolder)
    except:
        pass

    try:
        os.mkdir(os.path.join(baseFolder,datasetName))
    except:
        pass



def get_strided_data(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    inp_no_start = inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]
    inp_std = inp_no_start.std(axis=(0, 1))
    inp_mean = inp_no_start.mean(axis=(0, 1))
    inp_norm=inp_no_start
    #inp_norm = (inp_no_start - inp_mean) / inp_std

    #vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    #inp_norm=np.concatenate((inp_norm,vis),2)

    return inp_norm[:,:gt_size-1],inp_norm[:,gt_size-1:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}


def get_strided_data_2(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    inp_relative_pos= inp_te_np-inp_te_np[:,:1,:]
    inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]),1)
    inp_accel = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_speed[:,1:,0:2] - inp_speed[:, :-1, 0:2]),1)
    #inp_std = inp_no_start.std(axis=(0, 1))
    #inp_mean = inp_no_start.mean(axis=(0, 1))
    #inp_norm= inp_no_start
    #inp_norm = (inp_no_start - inp_mean) / inp_std

    #vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    #inp_norm=np.concatenate((inp_norm,vis),2)
    inp_norm=np.concatenate((inp_te_np,inp_relative_pos,inp_speed,inp_accel),2)
    inp_mean=np.zeros(8)
    inp_std=np.ones(8)

    return inp_norm[:,:gt_size],inp_norm[:,gt_size:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}

def get_strided_data_clust(dt, gt_size, horizon, step):
    inp_te = []
    dtt = dt.astype(np.float32)
    raw_data = dtt

    ped = raw_data.ped.unique()
    frame=[]
    ped_ids=[]
    for p in ped:
        for i in range(1+(raw_data[raw_data.ped == p].shape[0] - gt_size - horizon) // step):
            frame.append(dt[dt.ped == p].iloc[i * step:i * step + gt_size + horizon, [0]].values.squeeze())
            # print("%i,%i,%i" % (i * 4, i * 4 + gt_size, i * 4 + gt_size + horizon))
            inp_te.append(raw_data[raw_data.ped == p].iloc[i * step:i * step + gt_size + horizon, 2:4].values)
            ped_ids.append(p)

    frames=np.stack(frame)
    inp_te_np = np.stack(inp_te)
    ped_ids=np.stack(ped_ids)

    #inp_relative_pos= inp_te_np-inp_te_np[:,:1,:]
    inp_speed = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_te_np[:,1:,0:2] - inp_te_np[:, :-1, 0:2]),1)
    #inp_accel = np.concatenate((np.zeros((inp_te_np.shape[0],1,2)),inp_speed[:,1:,0:2] - inp_speed[:, :-1, 0:2]),1)
    #inp_std = inp_no_start.std(axis=(0, 1))
    #inp_mean = inp_no_start.mean(axis=(0, 1))
    #inp_norm= inp_no_start
    #inp_norm = (inp_no_start - inp_mean) / inp_std

    #vis=inp_te_np[:,1:,2:4]/np.linalg.norm(inp_te_np[:,1:,2:4],2,axis=2)[:,:,np.newaxis]
    #inp_norm=np.concatenate((inp_norm,vis),2)
    inp_norm=np.concatenate((inp_te_np,inp_speed),2)
    inp_mean=np.zeros(4)
    inp_std=np.ones(4)

    return inp_norm[:,:gt_size],inp_norm[:,gt_size:],{'mean': inp_mean, 'std': inp_std, 'seq_start': inp_te_np[:, 0:1, :].copy(),'frames':frames,'peds':ped_ids}


def distance_metrics(gt,preds,dim_3 =False):
    errors = np.zeros(gt.shape[:-1])
    if dim_3:
        for i in range(errors.shape[0]):
            for j in range(errors.shape[1]):
                for k in range(errors.shape[2]):
                    errors[i, j, k] = scipy.spatial.distance.euclidean(gt[i, j , k], preds[i, j,k])
        return errors.mean(),errors[:,:,-1].mean(),errors
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
    return errors.mean(),errors[:,-1].mean(),errors

def load_datasets(args,add_c=True):


            ## creation of the dataloaders for train and validation
    if args.val_size==0:   # validation size is zero
        train_dataset,_ = create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True,verbose=args.verbose)
        val_dataset, _ = create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs,
                                                                    args.preds, delim=args.delim, train=False,
                                                                    verbose=args.verbose)
    else:
        train_dataset, val_dataset = create_dataset(args.dataset_folder, args.dataset_name, args.val_size,args.obs,
                                                              args.preds, delim=args.delim, train=True,
                                                              verbose=args.verbose)

    test_dataset,_ =  create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)
    
    means=[]
    stds=[]
    for i in np.unique(train_dataset[:]['dataset']):
        ind=train_dataset[:]['dataset']==i
        means.append(torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).mean((0, 1)))
        stds.append(torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).std((0, 1)))

    mean=torch.stack(means).mean(0)
    std=torch.stack(stds).mean(0)

    
    return train_dataset, val_dataset,test_dataset,mean,std

if __name__ == '__main__':
        ## creation of the dataloaders for train and validation
    if Args.val_size==0:   # validation size is zero
        train_dataset,_ = create_dataset(Args.dataset_folder,Args.dataset_name,0,Args.obs,Args.preds,delim=Args.delim,train=True,verbose=Args.verbose)
        val_dataset, _ = create_dataset(Args.dataset_folder, Args.dataset_name, 0, Args.obs,
                                                                    Args.preds, delim=Args.delim, train=False,
                                                                    verbose=Args.verbose)
    else:
        train_dataset, val_dataset = create_dataset(Args.dataset_folder, Args.dataset_name, Args.val_size,Args.obs,
                                                              Args.preds, delim=Args.delim, train=True,
                                                              verbose=Args.verbose)

    test_dataset,_ =  create_dataset(Args.dataset_folder,Args.dataset_name,0,Args.obs,Args.preds,delim=Args.delim,train=False,eval=True,verbose=Args.verbose)
    
    print("train_dataset: ",len(train_dataset))
    print("val_dataset: ",len(val_dataset))
    print("test_dataset: ",len(test_dataset))
    un = np.unique(train_dataset[:]['dataset'])
    means=[]
    stds=[]
    for i in np.unique(train_dataset[:]['dataset']):
        print(" Unique Numbers: ",i, " UN: ",un)
        ind=train_dataset[:]['dataset']==i
        means.append(torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).mean((0, 1)))
        stds.append(torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).std((0, 1)))
    print(" Means: ",means)
    mean=torch.stack(means)#.mean(0)
    print(" Means stacked: ",mean)
    mean=torch.stack(means).mean(0)
    std=torch.stack(stds).mean(0)
    print(f"Mean {mean} std: {std}")

    # #sas = train_dataset[0]['src'][False, 1:, 2:4]
    # src = train_dataset[0]['src'][ 1:, 2:4]
    # trg = train_dataset[0]['trg'][ 1:, 2:4]
    # trg2 = train_dataset[:]['trg'][ 1:, 2:4]
    # target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
    device = CFG.device
    i = 0
    for id_b,batch in enumerate(tr_dl):

            inp=(batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
            target=(batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
            target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
            #print("target_c: ",target_c.shape)
            target=torch.cat((target,target_c),-1)
            #print("target_c n target: ",target.shape)
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)
            dec_inp = torch.cat((start_of_seq, target), 1)
            src_att = torch.ones((inp.shape[0], 1,inp.shape[1])).to(device)
            trg_att=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)

            trail = torch.sqrt(torch.square(batch['src'][:,:,2].to(device)) + torch.square(batch['src'][:,:,3].to(device))).unsqueeze(-1)
            print("trail: ",trail.shape)
            print("trail: ",trail[0])
            newput = torch.cat((batch['src'].to(device),trail),-1)
            #trg_att=subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device)
            # print("inp with mean: ",inp[0])
            # print("inp: ",batch['src'][:,:,:][0])
            # print("src_att: ",src_att.shape)
            # #print("src: ",src_att)
            # print("start_of_seq: ",start_of_seq.shape)
            # print("trg_att shape: ",trg_att.shape)
            # print("inp: ",inp.shape)
            # print("dec_inp: ",dec_inp.shape)
            # print("dec_inp :",dec_inp[0]," \n //////////////////////////////////////////")
            # dec_inp
            print("SRC:",batch['src'][0])
            print("SRC:",newput[0])

            #print("target_c: ",target_c)
            break

            # if i==1:
            #     break
            # i+=1
            
    # ind  = True
    # cated = torch.cat((train_dataset[0]['src'][ind, 1:, 2:4], train_dataset[0]['trg'][ind, :, 2:4]), 1)
    # cated2 = torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1)
    
    # #print("SAS: ",sas)
    # print("src: ",src)
    # print("trg: ",trg)
    # print("cated: ",cated)
    # print("cated2: ",cated2.shape)
    # print("cated2: ",cated2[0].shape)
    # print("SAMPLE SRC: ",(train_dataset[:]['src'][ind, 1:, 2:4]).shape)
    # print("SAMPLE TRG: ",(train_dataset[:]['trg'][ind, :, 2:4]).shape)
    # print("last trg: ",train_dataset[30306]['trg'][ind, :, 2:4])
    # print("last src: ",train_dataset[30306]['src'][ind, 1:, 2:4])
    # print("last trg  7: ",train_dataset[30307]['trg'][ind, :, 2:4])
    # print("last src  7: ",train_dataset[30307]['src'][ind, 1:, 2:4])
    # print("test sample: ",test_dataset[0])
    # print("test sample src: ",test_dataset[0]['src'])
    # print("test sample trg: ",test_dataset[0]['trg'])
    # print("test sample seq_start: ",test_dataset[0]['seq_start'])
    # print("test sample dataset: ",val_dataset[5410]['dataset'])
    # print("test sample peds: ",test_dataset[0]['peds'])
    # print("test sample peds: ",test_dataset[1]['peds'])
    # print("test sample peds: ",test_dataset[10]['peds'])