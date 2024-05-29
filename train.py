import torch
from models.options import ParamOptions
from models.trainer import TrainModel
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    opt = ParamOptions().parse()
    model = TrainModel(opt)
#     model.init_model()
    step, step_test, save_name = model.init_model(opt.model_A, opt.model_B)
    writer = SummaryWriter()
    init_epoch = 0
    
    if opt.load_weights.lower() == 'true':
        # print("loading saved model...")
        init_epoch = int(input('Input the initial epoch:'))
        init_time = input('Input the corresponding date (ex: Jun04_19_31_{}_{}, {}: UNet, WNet, SRResNet):')
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
    
    for epoch in range(opt.num_epochs):
        model.update_learning_rate(epoch)
        
        train_loss = torch.zeros(len(model.train_loader), 4)
        val_loss = torch.zeros(len(model.test_loader), 4)
        
        loss_D_A = 0
        loss_G_A = 0
        loss_D_B = 0
        loss_G_B = 0
        
        # print("training the  model...")
        for i, train_data in enumerate(model.train_loader):
            model.set_input(train_data)
            model.optimization('train')
            losses = model.get_current_losses('train')
            
            writer.add_scalar('data/train_D_A', losses['D_A'], epoch * len(model.train_loader) + i)
            writer.add_scalar('data/train_G_A', losses['G_A'], epoch * len(model.train_loader) + i)
            writer.add_scalar('data/train_D_B', losses['D_B'], epoch * len(model.train_loader) + i)
            writer.add_scalar('data/train_G_B', losses['G_B'], epoch * len(model.train_loader) + i)
            
            loss_D_A += losses['D_A']
            loss_G_A += losses['G_A']
            loss_D_B += losses['D_B']
            loss_G_B += losses['G_B']
            
            train_loss[i, 0] = losses['D_A']
            train_loss[i, 1] = losses['G_A']
            train_loss[i, 2] = losses['D_B']
            train_loss[i, 3] = losses['G_B']
            
            model.print_current_losses(epoch, i, losses)

            if i % opt.print_loss_freq_iter == opt.print_loss_freq_iter -1:
                losses = model.get_current_losses()
                model.print_current_losses(epoch,i,losses)
                
            if i % opt.save_cycleplot_freq_iter == opt.save_cycleplot_freq_iter -1:
                model.visual_iter(epoch,i)
                
        if epoch % opt.save_val_freq_epoch == opt.save_val_freq_epoch -1:
#                     print("validating the model...")
            val_loss_D_A = 0
            val_loss_G_A = 0
            val_loss_D_B = 0
            val_loss_G_B = 0
            with torch.no_grad():
                for k,test_data in enumerate(model.test_loader):
                    model.val_input(test_data)
                    model.forward()
                    model.visual_val(epoch,k)
                    
                    writer.add_scalar('data/val_D_A', val_losses['D_A_test'], 
                                      epoch * len(model.test_loader) + k)
                    writer.add_scalar('data/val_G_A', val_losses['G_A_test'], 
                                      epoch * len(model.test_loader) + k)
                    writer.add_scalar('data/val_D_B', val_losses['D_B_test'], 
                                      epoch * len(model.test_loader) + k)
                    writer.add_scalar('data/val_G_B', val_losses['G_B_test'], 
                                      epoch * len(model.test_loader) + k)
                    
                    val_loss_D_A += val_losses['D_A_test']
                    val_loss_G_A += val_losses['G_A_test']
                    val_loss_D_B += val_losses['D_B_test']
                    val_loss_G_B += val_losses['G_B_test']
                    
                    val_loss[k, 0] = val_losses['D_A_test']
                    val_loss[k, 1] = val_losses['G_A_test']
                    val_loss[k, 2] = val_losses['D_B_test']
                    val_loss[k, 3] = val_losses['G_B_test']
                    
                    model.print_current_losses(epoch, k, val_losses)
            
            writer.add_scalars('data_per_epoch/D_A', 
                               {'val': val_loss_D_A / len(model.test_loader)}, epoch)
            writer.add_scalars('data_per_epoch/G_A', 
                               {'val': val_loss_G_A / len(model.test_loader)}, epoch)
            writer.add_scalars('data_per_epoch/D_B', 
                               {'val': val_loss_D_B / len(model.test_loader)}, epoch)
            writer.add_scalars('data_per_epoch/G_B', 
                               {'val': val_loss_G_B / len(model.test_loader)}, epoch)
            
        if epoch % opt.save_model_freq_epoch == opt.save_model_freq_epoch -1:
            model.save_models(epoch, 'test')
            writer.export_scalars_to_json(f"training_and_validation_curves/all_scalars_epoch{epoch+1}_{opt.model_A}_{opt.model_B}_{save_name}.json")
    
    writer.close()
