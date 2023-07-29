import torch
import torch.nn as nn
import cv2
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from baselineUtils import load_datasets,distance_metrics
from utils import ScheduledOptim,visualize_preds,visualize_preds_only
from utils import generate_square_mask
from trajectory_candidates import run_cluster
from config import CFG ,Args,cfg_zed
from utils import person
from train import train_attn_mdn 
from model import Attention_GMM

#################################################################   ZED CAMERA ###############################################################
def train(train_dl,val_dl,test_dl):
    # Train
    loss_type = Args.loss_mode
    in_features = CFG.in_features
    out_features = CFG.out_features
    num_heads = CFG.num_heads
    num_encoder_layers = CFG.num_encoder_layers
    num_decoder_layers =  CFG.num_decoder_layers
    embedding_size = CFG.embd_size
    max_length = 8
    n_hidden = CFG.n_hidden
    gaussians = CFG.gaussians
    forecast_window = 12
    drp = CFG.drop_out
    add_features = CFG.add_features
    print("The following parameters were used for Training:")
    # Create a list of (variable, value) pairs
    variables = [
        ('loss_type', loss_type),
        ('in_features', in_features),
        ('out_features', out_features),
        ('num_heads', num_heads),
        ('num_encoder_layers', num_encoder_layers),
        ('num_decoder_layers', num_decoder_layers),
        ('embedding_size', embedding_size),
        ('max_length', max_length),
        ('n_hidden', n_hidden),
        ('gaussians', gaussians),
        ('forecast_window', forecast_window),
        ('drp', drp),
        ('add_features', add_features)
    ]

    # Construct the output string
    output_string = '\n'.join(f'{var} = {val}' for var, val in variables)

    # Print the output string
    print(output_string)
    print("\nTraining parameters can be changed from config.py\n")
    # If you want to train transformer only copy it from commented.py
    # Train attention MDN
    #attn_mdn = Attention_GMM(device,in_features,out_features,num_heads,num_encoder_layers,num_decoder_layers,embedding_size,n_gaussians=gaussians,n_hidden = n_hidden, dropout=drp).to(device)
    attn_mdn = Attention_GMM(in_features,out_features,num_heads,num_encoder_layers,num_decoder_layers,embedding_size,n_gaussians=gaussians,n_hidden = n_hidden, dropout=drp).to(device)
    #attn_mdn = Attention_GMM_Encoder(device,in_features,out_features,num_heads,num_encoder_layers,num_decoder_layers,embedding_size,n_gaussians=gaussians,n_hidden = n_hidden, dropout=drp).to(device)
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs")
    #     attn_mdn = DDP(attn_mdn,device_ids=[0,1])
    for p in attn_mdn.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # Define the optimizer
    optimizer = ScheduledOptim(
            torch.optim.Adam(attn_mdn.parameters(), betas=(0.9, 0.98), eps=1e-09),
            CFG.lr_mul, CFG.d_model, CFG.n_warmup_steps) #len(train_dl)True
    #         print(name, child)
    loss_train, loss_eval,val_mad,val_fad = train_attn_mdn(train_dl,val_dl,test_dl,attn_mdn,optimizer,add_features,mixtures =gaussians, epochs=CFG.epochs,mean=mean,std=std)

    # Results
    fig = plt.figure(1)	#identifies the figure 
    plt.title(" Training Loss Per Epoch", fontsize='16')	#title
    plt.plot(loss_train,color='Blue',label='Training Loss')	#plot the points
    plt.plot(loss_eval,color='Green',label='Evaluation Loss')	#plot the points
    plt.legend(loc="upper right")
    plt.show()
    fig = plt.figure(2)	#identifies the figure 
    plt.title("Evaluation Error", fontsize='16')	#title
    # plt.plot(test_mad,color='Green', label="ADE Test")	#plot the points
    # plt.plot(test_fad,color='Red', label="FDE Test")	#plot the points
    plt.plot(val_mad,color='Green', label="ADE Validation")	#plot the points
    plt.plot(val_fad,color='Red', label="FDE Validation")	#plot the points
    plt.legend(loc="upper right")
    plt.show()

    return attn_mdn

def infer_real_time(attn_mdn,device,add_features,mean,std,pf,rate=cfg_zed.data_rate,normalized=True,enc_seq = 8,dec_seq=12):
    import pyzed.sl as sl
    # Create a Camera object
    zed = sl.Camera()
    zed_pose = sl.Pose()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode    
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_fps = cfg_zed.frame_rate #cfg_zed['frame_rate']                          # Set fps at 30
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    #camera_position = zed.get_position(zed_pose, sl.REFERENCE_FRAME.CAMERA)

    # Set runtime parameters
    runtime_parameters = sl.RuntimeParameters()

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable object detection module
    obj_param = sl.ObjectDetectionParameters()
    # Defines if the object detection will track objects across images flow.
    obj_param.enable_tracking = True       # if True, enable positional tracking

    if obj_param.enable_tracking:
        zed.enable_positional_tracking()
        
    zed.enable_object_detection(obj_param)

    camera_info = zed.get_camera_information()
    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    # viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking)

    # Configure object detection runtime parameters
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 60
    obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.PERSON]    # Only detect Persons

    # Create ZED objects filled in the main loop
    objects = sl.Objects()
    image = sl.Mat()
    objects_dic= dict()          # Dictionary holding objects values 
    frame = 0  # Frame number
    # For long horizion prediction we should sample every frame divisible by frame_sample
    frame_sample = cfg_zed.frame_rate/ cfg_zed.data_rate  # frame sample tells you either to take info in frame or not     
    while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:#viewer.is_available():
        # Grab an image, a RuntimeParameters object must be given to grab()
        # if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        image_ocv = image.get_data() 
        # Retrieve objects
        zed.retrieve_objects(objects, obj_runtime_param)
        batch_dic=dict()    
        input_list =[] 
        bbox_list =[]
        dists = []
        object_added = False 
        frame+=1
        print("Mod: ",frame%frame_sample) 
        # if (objects.is_new) :
        #     obj_detected= True   
        # else:
        #     obj_detected= False
        if (objects.is_new):
            
            # Count the number of objects detected
            if (pf):
                print("{} Object(s) detected".format(len(objects.object_list)))
            if len(objects.object_list):
                #obj_detected= False
                
                for obj in objects.object_list:
                    #obj_detected= True
                    distance = math.sqrt(obj.position[0]*obj.position[0] + obj.position[2]*obj.position[2])
                    box = [obj.position[0],obj.position[2],0,0]
                    height = obj.position[1]
                    bounding_box = obj.bounding_box_2d 
                    frame_rate = zed.get_camera_information().camera_fps
                    bbox = np.concatenate((bounding_box[0], bounding_box[2]), axis=0)
                    #print("bboxbboxbboxbboxbbox: ",bbox)
                    start_point = (int(bbox[0]),int(bbox[1]))
                    end_point = (int(bbox[2]),int(bbox[3]))
                    #cv2.rectangle(image_ocv, start_point, end_point,cfg_zed.crossing_color, cfg_zed.line_thickness)


                    key = obj.id
                    velocity_is_nan = math.isnan(obj.velocity[0])

                    if key in objects_dic: 
                        object_added = True
                        if ((frame%frame_sample)==0):
                            frame=0
                            objects_dic[key].add_box(box)
                            objects_dic[key].bbox = bbox
                            batch_dic[key] =  objects_dic[key]
                        input_list.append(objects_dic[key].history)
                        bbox_list.append(bbox)
                        dists.append(distance)
                    else:
                        new_person = person(key,box,bbox,distance,height,frame_rate)
                        batch_dic[key] = new_person
                    # elif (not velocity_is_nan):
                    #     new_person = person(key,box,bbox,distance,height,frame_rate)
                    #     batch_dic[key] = new_person
                
                objects_dic = batch_dic
        input_dataset = torch.Tensor(np.array(input_list))
        if object_added:#object_added:
            if pf:
                print("shape:",len(input_dataset),len(input_dataset[0]),len(input_dataset[0][0]))
                print("bbox_list:",len(bbox_list),len(bbox_list[0]))
        #test_dl = torch.utils.data.DataLoader(input_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
            with torch.no_grad():
                attn_mdn.eval()
                
                #input_valc = torch.sqrt(torch.square(input_dataset[:,1:,2].to(device)) + torch.square(input_dataset[:,1:,3].to(device))).unsqueeze(-1)
            # for id_e, val_batch in enumerate(test_dl):
            #     pass
                # inp_val=(val_batch[:,1:,2:4].to(device)-mean.to(device))/std.to(device)
                # input_valc = torch.sqrt(torch.square(val_batch[:,1:,2].to(device)) + torch.square(val_batch[:,1:,3].to(device))).unsqueeze(-1)
                if normalized:
                    inp_val=(input_dataset[:,1:,2:4].to(device)-mean.to(device))/std.to(device)
                    
                
                else :
                    inp_val = input_dataset[:,1:,2:4].to(device)

                if (add_features):                
                    input_valc = torch.sqrt(torch.square(input_dataset[:,1:,2].to(device)) + torch.square(input_dataset[:,1:,3].to(device))).unsqueeze(-1)
                    input_val = torch.cat((inp_val,input_valc),-1)
                else:
                    input_val = inp_val
                tgt_val = torch.Tensor([0, 0]).unsqueeze(0).unsqueeze(1).repeat(input_val.shape[0],12,1).to(device)
                tgt_val_mask = generate_square_mask(dim_trg = dec_seq ,dim_src = enc_seq, mask_type="tgt").to(device)
                pi, sigma_x,sigma_y, mu_x , mu_y,decoder_out = attn_mdn(input_val,tgt_val,tgt_mask = tgt_val_mask)
                mus = torch.cat((mu_x.unsqueeze(-1),mu_y.unsqueeze(-1)),-1)
                #sigmas = torch.cat((sigma_x.unsqueeze(-1),sigma_y.unsqueeze(-1)),-1)

                src_value = input_dataset[:, -1:,0:2].detach().cpu().numpy()
                src_value = src_value[:,np.newaxis,:,:]
                cluster_mus = (mus[:, :,:] * std.to(device) + mean.to(device)).detach().cpu().numpy().cumsum(1) + src_value

                # cluster_real = val_batch['trg'][:, :, 0:2]
                cluster_real = input_dataset[:, :, 0:2]
                cluster_src = input_dataset[:,:,0:2]
                batch_trajs,batch_weights,best_trajs,best_weights = run_cluster(cluster_mus,pi,cluster_real,cluster_src,dbscan=False)
                if pf:
                    print("best_candiates: ",best_trajs,best_weights)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            for trajs,traj,bbox,wgt,dist in zip (batch_trajs,best_trajs,bbox_list,best_weights,dists):
                start_point = (int(bbox[0]),int(bbox[1]))
                end_point = (int(bbox[2]),int(bbox[3]))
                # half_w = int(int(bbox[0])+int(bbox[2])/2)
                # half_h = int(int(bbox[1])+int(bbox[3])/2)
                # org = (half_w, half_h)
                count = sum(n[0] < 0 for n in traj)
                # print("count: ",count)
                # print("traj: ",traj)
                if wgt>0.99:
                    wgt = 0.98
                kind = str(int(wgt*100)) + "% Safe!"
                if (dist < cfg_zed.safe_distance):
                    if pf:
                        print("Dangerously Close to Vehicle!!")
                    cv2.rectangle(image_ocv, start_point, end_point,cfg_zed.danger_color, cfg_zed.line_thickness)
                    kind = str(int(wgt*100)) + "% DANGER!"
                elif (count!=0 and count!=12)and (dist < cfg_zed.safe_distance_lvl2) :
                     if pf:
                        print("Crossing Red!",dist)
                     cv2.rectangle(image_ocv, start_point, end_point,cfg_zed.crossing_color, cfg_zed.line_thickness)
                     kind = str(int(wgt*100)) + "% Crossing Red!"
                elif (count!=0 and count!=12):# and (dist < cfg_zed.max_cross_detection)) :
                     if pf:
                        print("Crossing Blue!",dist)
                     cv2.rectangle(image_ocv, start_point, end_point,cfg_zed.crossing_color, cfg_zed.line_thickness)
                     kind = str(int(wgt*100)) + "% Crossing Blue!!"
                else:
                     if pf:
                        print("SAFE")
                     cv2.rectangle(image_ocv, start_point, end_point,cfg_zed.safe_color, cfg_zed.line_thickness) 

                #cv2.rectangle(image_ocv, start_point, end_point,cfg_zed.crossing_color, cfg_zed.line_thickness)
                # Using cv2.putText() method
                cv2.putText(image_ocv, kind, start_point, font, 
                                fontScale, cfg_zed.crossing_color, cfg_zed.line_thickness, cv2.LINE_AA)
                             
        cv2.imshow("Image", image_ocv)
        cv2.waitKey(1)
    image.free(memory_type=sl.MEM.CPU)
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    cv2.destroyAllWindows()
    zed.close()


def infer_simulation(attn_mdn,device,add_features,mean,std,pf,data_rate=cfg_zed.data_rate,normalized=True,enc_seq = 8,dec_seq=12):
    from pedestrian_input import PedestrianPast,MarkerBase
    import rospy
    rospy.init_node('pedestrian_marker')
    points = 8
    frequency = 2
    pedestrians = []
    pedestrians.append(PedestrianPast("actor1", points))
    pedestrians.append(PedestrianPast("actor2", points))

    Base = MarkerBase(pedestrians, frequency, points)
    Base.predictor(attn_mdn,device,add_features,mean,std,pf,data_rate=cfg_zed.data_rate,normalized=True,enc_seq = 8,dec_seq=12)
    Base.start()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs Testing/Inference based on mode!')
    parser.add_argument('-md','--mode',default=Args.mode,help='Enable mode ->train/test/inference/simulation')
    parser.add_argument('-rt','--real-time', action='store_true', default=Args.real_time,help='Enable Inference mode real_time/dataset -> inference no real time means normal testing without error output (Faster)')
    parser.add_argument('-mpt','--model_path',default=Args.model_path,help='trained model path')
    parser.add_argument('-d','--device',default=CFG.device,help='Cuda Device')
    parser.add_argument('-af','--add_features',action='store_true',default=CFG.add_features,help='add handcrafted features')
    parser.add_argument('-pf','--pformat',action='store_true',help='inference_print_format -> if activated will print details')


    mean = torch.tensor([-6.5713e-03, -8.1075e-05])
    std = torch.tensor([0.3120, 0.1511])
    

    args = parser.parse_args()

    print("\n-----Running-----")
    # Iterate over the arguments
    for arg_name, arg_value in vars(args).items():
        print(arg_name, '=', arg_value)
    print("-----------------\n")

    if ((args.mode=='test') or (args.mode=='train')) :
        print("Preparing Dataset ...\n")
        dataset_args = Args 
        train_dataset, val_dataset,test_dataset,mean,std = load_datasets(dataset_args)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
        val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

        if(args.mode=='train'):
            print("Preparing Model For Training ...")
    
           #attn_mdn = train(train_dl,val_dl,test_dl)
    else:
        print("Loading Model For Inference ...")
        PATH = args.model_path
        print("MODEL PATH: ",PATH)
        attn_mdn = torch.load(PATH).to(args.device)
        print("Model Loaded!")
    
    # Count the number of parameters
    num_params = sum(p.numel() for p in attn_mdn.parameters() if p.requires_grad)
    print(f"The model has {num_params} parameters.")

    if (args.mode=='simulation'):
        infer_simulation(attn_mdn,args.device,args.add_features,mean,std,args.pformat)
    elif args.real_time:
        print("REAL_TIME PREDICTION")
        infer_real_time(attn_mdn,args.device,args.add_features,mean,std,args.pformat)
    