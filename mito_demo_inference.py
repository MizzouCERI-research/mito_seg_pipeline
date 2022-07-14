# %%
# region IMPORT ###########################################
###########################################################
import numpy as np
import torch
import torch.nn.functional as Fn
try:
    from torchinfo import summary
except: 
    from torchsummary import summary

# import warnings 
# warnings.filterwarnings("ignore")

from src.data.data_demo_r import get_data
from src.inference.aggregator import GridAggregator
from src.inference.grid_sampler import GridSampler
from connectomics.utils.processing import bc_watershed # after installing
import hdf5storage
from src.metrics.detection3d.detection3d_eval import get_mAP_3d
# endregion IMPORT





# %%
# region DATA #############################################
###########################################################
patch_size_train = (224,224,80)
patch_size_val = (224,224,80)
patch_size_test = (224,224,80)
patch_data = get_data(data_root='data/1k', batch_size=(1,1,1,1), num_patches=(1800,500,0,0), patch_size=(patch_size_train, patch_size_val, patch_size_test))

train_dataset = patch_data['train_dataset']
val_dataset = patch_data['val_dataset']
eval_dataset = patch_data['eval_dataset']  
test_dataset = patch_data['test_dataset']
train_loader = patch_data['train_loader'] 
val_loader = patch_data['val_loader']
eval_loader = patch_data['eval_loader']
test_loader = patch_data['test_loader'] 
# endregion DATA




# %%
# region MODEL ############################################ 
###########################################################


device = torch.device("cuda:0") # all devices
checkpoint = torch.load('checkpoints/mito_rat.pth', map_location=device)
model = checkpoint['model_state_dict']
optimizer = checkpoint['optimizer_state_dict']
del checkpoint


# %%
inputs = torch.randn(1,1,224,224,80).to(device)
outputs = model(inputs)
summary(model, input_data=inputs, depth=4, col_names=['output_size', 'num_params'])



# %%
# LOSS ---------------------
# --------------------------


BASE_NUM_KERNELS = 64
EPS = 1e-9

def dice(prediction, truth):
    return 2.0 * torch.sum(prediction * truth, dim=[1, 2, 3, 4]) / (torch.sum((prediction ** 2 + truth ** 2), [1, 2, 3, 4]) + EPS)

def dice_2(prediction, truth):
    return 2.0 * torch.sum(prediction * truth, dim=[0, 1, 2]) / (torch.sum((prediction ** 2 + truth ** 2), [0, 1, 2]) + EPS)



def dice_score(prediction, truth):

    if len(prediction.shape) > 4:
        dc = dice(prediction, truth)
    else:
        dc = dice_2(prediction, truth)
    return torch.mean(dc, dim=0)


# endregion MODEL




# %%
# region TEST ############################################
##########################################################



patch_overlap = (16, 16, 16)
test_batch = 2
label_channels = 2


dscores = []

test_outputs = torch.tensor([], dtype=torch.float32, device=torch.device('cpu'))
test_outputs_thres = torch.tensor([], dtype=torch.float32, device=torch.device('cpu'))

num_tests = len(eval_loader)

model.eval()
print('')



# %%
with torch.no_grad():
    for batch_idx, batch_data in enumerate(eval_loader):
        if batch_idx in list(range(16)):
            batch_image = batch_data['image']
            batch_target = batch_data['label']
            batch_file_name = batch_data['file_name']
                
            
            for i in range(batch_image.shape[0]):

                input_tensor = batch_image[i][0]

                grid_sampler = GridSampler(
                input_tensor,
                patch_size_test,
                patch_overlap,
                )
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=test_batch)
                aggregator = GridAggregator(grid_sampler)

                with torch.no_grad():
                    for patches_batch in patch_loader:
                        inputs = patches_batch['image'].to(device)
                        locations = patches_batch['location']
                        logits = model(inputs)
                        labels = Fn.sigmoid(logits)
                        aggregator.add_batch(labels, locations)

                
                foreground = aggregator.get_output_tensor()

                output_thres = torch.zeros_like(foreground)
                output_thres[foreground>0.5] = 1

                tem_name = 'temp/jpg/' + batch_file_name[i] + '.jpg'

            test_outputs = torch.cat((test_outputs, foreground.cpu().unsqueeze(dim=0)), dim=0)
            test_outputs_thres = torch.cat((test_outputs_thres,  output_thres.cpu().unsqueeze(dim=0)), dim=0)

            print(f'Batch {batch_idx+1}/{num_tests}')


# %%
del batch_data
del foreground


# # %%
# plot_volume_interactive(test_outputs_thres[0,0].numpy())

# # %%

# plot_volume_interactive(test_outputs_thres[0,1].numpy())

# # %%
# plot_volume_interactive(eval_dataset.data[0]['label'][0])





# %%
print('====================Semenatic Segmentation====================')
for i in range(len(eval_loader)):
    dice_index = dice_score(test_outputs_thres[i,0], torch.tensor(eval_dataset.data[i]['label'][0]))
    print(f'Rat eval volume {i}, Dice score = {dice_index}')
print('\n\n') 



# %%
print('====================Instance Segmentation====================')
for i in range(len(eval_loader)):
    print(f'---------Volume {i}---------')
    pred_instances = bc_watershed(test_outputs[i].numpy()*255, thres1=0.5, thres2=0.8, thres3=0.1, thres_small=128)
    get_mAP_3d(gt_seg=eval_dataset.data[i]['instances'][0], pred_seg=pred_instances, predict_heatmap=test_outputs[i][0])

# %%
    pred_instances_dict = {'pred_instances': pred_instances.astype(np.uint16)}
    hdf5storage.savemat('./' + f'pred_instances_rat_{i}.mat', pred_instances_dict, format='7.3') 


# %%
    semantic_dict = {'pred_semantic': test_outputs_thres[i].numpy().astype(np.uint8)}
    hdf5storage.savemat('./' + f'pred_semantic_rat_{i}.mat', semantic_dict, format='7.3')
    print('\n\n') 




# endregion TEST







# %%
# region MAIN #############################################
###########################################################


# if __name__ == "__main__":

#     train()

#     # state_dict_file = './model.pth'
#     # model.load_state_dict(torch.load(state_dict_file))
#     test()

# # %%

# endregion MAIN



