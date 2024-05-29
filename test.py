import torch
from models.options import ParamOptions
from models.trainer import TrainModel
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    opt = ParamOptions().parse()
    model = TrainModel(opt)
    step, step_test, save_name = model.init_model(opt.model_A, opt.model_B)
    writer = SummaryWriter()
    init_epoch = 0
    
    init_epoch = int(input('Input the initial epoch:'))
    init_time = input('Input the corresponding date (ex: Jun04_19_31_{}_{}, {}: UNet):')
    save_name = init_time
    path = f'pretrained_model/{init_time}/save/' 
    model.load_models(
        model, 
        str(init_epoch), 
        path, 
        f'{init_time}_{opt.model_A}_{opt.model_B}_netG_A_{init_epoch}ep.pt', 
        f'{init_time}_{opt.model_A}_{opt.model_B}_netD_A_{init_epoch}ep.pt', 
        f'{init_time}_{opt.model_A}_{opt.model_B}_netG_B_{init_epoch}ep.pt', 
        f'{init_time}_{opt.model_A}_{opt.model_B}_netD_B_{init_epoch}ep.pt'
    )
    
    with torch.no_grad():
        for k, test_data in enumerate(model.test_loader):
            model.set_input(test_data)
            model.optimization('test')
            model.visual_val(init_epoch, k, opt.model_A, opt.model_B)
            val_losses = model.get_current_losses('test')
            model.print_current_losses(init_epoch, k, val_losses)
