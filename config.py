import torch

class Args:  # Dataset areguments
    val_size = 0   # validation dataset needed
    dataset_folder = 'datasets'#'datasets'#'sample'#
    dataset_name = 'zara2'#'univ' # #'zara2'#'zara1'#'eth'  # could be zara or others as well
    obs = 8
    preds = 12
    verbose = 2
    delim = '\t'
    mode = 'test'  # 'train'
    model_path = "saved_models/model_II_zara2_54_32_70_Final.pt"#model_II_zara2_54_32_70_Final.pt"
    visualize = False # to visualsize output

class CFG:
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # muti_gpu = True
    PATH = "saved_models/model_III_05_" + Args.dataset_name + "_"
    batch_size = 100
    num_encoder_layers = 6
    num_decoder_layers = 6
    num_heads = 8
    n_steps = 10
    n_warmup_steps = 3500
    lr_mul = 0.1
    epochs = 120
    zero_feed = False
    embd_size = 512
    n_hidden = 64
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

