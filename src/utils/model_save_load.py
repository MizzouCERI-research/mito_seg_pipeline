
import os
import torch




def save_model(model, device, model_dir='models/dicts', model_file_name='classifier.pt'):
    

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # make sure you transfer the model to cpu.
    if device == 'cuda':
        model.to('cpu')

    # save the state_dict
    torch.save(model.state_dict(), model_path)
    
    if device == 'cuda':
        model.to('cuda')
    
    return


def save_model_2(model, device, model_dir='models/dicts', model_file_name='classifier.pt'):
    

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # make sure you transfer the model to cpu.
    if device == 'cuda':
        torch.save(model.module.state_dict(), model_path)
    else:
        # save the state_dict
        torch.save(model.state_dict(), model_path)

    return


def save_model_3(model_name, model, optimizer):
    model_file = 'checkpoints/' + model_name + '.pth'
    torch.save({
                'model_state_dict': model,
                'optimizer_state_dict': optimizer,
                }, 
                model_file)




# state_dict_file_3d = 'models/net_2_detection_3d_new.pth'
# state_dict = torch.load(state_dict_file_3d)
# model_3d.load_state_dict(state_dict)

# check_file = 'save_3/net_2/net_2_detection_3d_train_monai_clstm_pos_dep3_epoch_34_metric_1222.5577_.pth'
# checkpoint = torch.load(check_file, map_location=device)
# model_3d = checkpoint['model_state_dict']
# optimizer = checkpoint['optimizer_state_dict']
# del checkpoint