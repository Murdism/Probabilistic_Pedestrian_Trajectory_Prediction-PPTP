# # Train transformer only
# in_features = 2
# out_features = 3
# num_heads = CFG.num_heads
# num_encoder_layers = CFG.num_encoder_layers
# num_decoder_layers =  CFG.num_decoder_layers
# embedding_size = CFG.embd_size
# max_length = 8
# n_hidden = 10
# n_gaussians = 5
# forecast_window = 12
# transformer_mdn = Transformer_MDN( device,in_features,out_features,num_heads,num_encoder_layers,num_decoder_layers,embedding_size).to(device)

# for p in transformer_mdn.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

# # Define the optimizer
# optimizer = ScheduledOptim(
#         torch.optim.Adam(transformer_mdn.parameters(), betas=(0.9, 0.98), eps=1e-09),
#         CFG.lr_mul, CFG.d_model, len(train_dl)*CFG.n_warmup_steps)
#mad , fad , loss = train_transformer_eth(train_dl,val_dl,transformer_mdn,optimizer,epochs=CFG.epochs,mean=mean,std=std)





# print("Train Error")
# gts , preds = predict_eth(train_dl,transformer_mdn,zero_feed=True,mean=mean,std=std) if CFG.zero_feed else predict_eth(train_dl,transformer_mdn,zero_feed=False,mean=mean,std=std)




# print("Test Error")
# gts , preds = predict_eth(test_dl,transformer_mdn,zero_feed=True,mean=mean,std=std) if CFG.zero_feed else predict_eth(test_dl,transformer_mdn,zero_feed=False,mean=mean,std=std)