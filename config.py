import torch
class ROS_PARAMS:
    ped_frequency = 2
    ped_num = 2
    ped_points = 8
class cfg_zed:
    safe_color =  [0, 255, 0]
    crossing_color = [255, 0, 0]
    danger_cross = [0, 125, 125]
    danger_color = [0, 0, 255]
    line_thickness = 2
    safe_distance = 1
    safe_distance_lvl2 = 2
    frame_rate = 15
    data_rate = 5 # rate at which position of pedestrian is rendered
    max_cross_detection = 15  # maximum distance where crossing path is relavant
class Args:  # Dataset areguments
    val_size = 0   # validation dataset needed
    dataset_folder = 'datasets'#'datasets'#'sample'#
    dataset_name = "zara2" # hotel'#'univ' # #'zara2'#'zara1'#'eth'  # could be zara or others as well
    obs = 8
    preds = 12
    verbose = 2
    delim = '\t'
    mode = 'simulation'#inference' #'test'  # 'train'
    real_time = True
    loss_mode = "mdn" # "msq","mdn","combined","pair_wise"
    model_path = "saved_models/model_V_zara2_71_H4_E3_032_070.pt"#"saved_models/model_II_zara2_54_32_70_Final.pt"#$"saved_models/model_V_zara1_47_H4_E3_042_092.pt"#"saved_models/model_II_zara2_54_32_70_Final.pt"#model_II_zara2_54_32_70_Final.pt"
    visualize = False # to visualsize output
    show_test = True # to show test results at every epoch during trainng
    

class CFG:
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # muti_gpu = True
    PATH = "saved_models/model_VI_" + Args.dataset_name + "_H4_E3_"
    
    batch_size = 100
    num_encoder_layers = 3  #6
    num_decoder_layers = 3  #6
    num_heads = 4   #8 
    n_steps = 10
    n_warmup_steps = 3500
    lr_mul = 0.1
    epochs = 100
    zero_feed = False
    embd_size = 128 #512
    n_hidden = 32 #64
    drop_out = 0.2
    d_model = embd_size
    gaussians = 8 #6
    num_workers = 8
    in_features = 3
    if in_features > 2:
        add_features = True
    else:
        add_features = False
    out_features = 2




"""
   train_dataset:  30,307    if val_size is x then train will be org_size - 7(x) --> 7 because we have 7 datasets(zara,eth,uni...)
   val_dataset:    5422
   test_dataset:   362
""" 
"""
V: parameters: > 28M
Heads :4
Encoders: 3
Decoders: 3
embd_size = 512
n_hidden = 64
"""


"""
VI: parameters: > 5M
Heads :4
Encoders: 3
Decoders: 3
embd_size = 128
n_hidden = 32
"""