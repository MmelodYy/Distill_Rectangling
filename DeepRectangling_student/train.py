import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import argparse
import skimage
from tqdm import tqdm
import numpy as np
from net.DistillModel import RectanglingNetwork
from torch.utils.data import DataLoader
from utils.dataSet import SPRectanglingTrainDataSet2TeachWeight,SPRectanglingTestDataSet
from net.loss_functions import Distill_loss_2Teacher_Weight
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.learningRateScheduler import warmUpLearningRate
import random
from thop import profile

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

setup_seed(2023)

def train_once(model,each_data_batch,epoch,epochs,criterion, optimizer,train_dataloaders):
    model.train()
    with tqdm(total=each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        each_batch_all_loss = 0
        each_batch_primary_loss = 0
        each_batch_super_loss = 0
        print("Start Train")
        for i, (img1,mask,lables,ds_mesh1,ds_mesh2,ds_weight_mesh1,ds_weight_mesh2) in enumerate(train_dataloaders):
            input_img = img1.float().cuda(device=args.device_ids[0])
            mask_img = mask.float().cuda(device=args.device_ids[0])
            lables = lables.float().cuda(device=args.device_ids[0])
            ds_mesh1 = ds_mesh1.float().cuda(device=args.device_ids[0])
            ds_mesh2 = ds_mesh2.float().cuda(device=args.device_ids[0])
            ds_weight_mesh1 = ds_weight_mesh1.float().cuda(device=args.device_ids[0])
            ds_weight_mesh2 = ds_weight_mesh2.float().cuda(device=args.device_ids[0])

            optimizer.zero_grad()
            mesh_final, warp_image_final, warp_mask_final = model.forward(input_img, mask_img)

            loss,primary_img_loss,super_img_loss = criterion(mesh_final, warp_image_final, warp_mask_final, ds_mesh1, ds_mesh2, ds_weight_mesh1, ds_weight_mesh2, lables)
            loss.backward()
            optimizer.step()

            each_batch_all_loss += loss.item() / args.train_batch_size
            each_batch_primary_loss += primary_img_loss.item() / args.train_batch_size
            each_batch_super_loss += super_img_loss.item() / args.train_batch_size
            pbar.set_postfix({'Lsum': each_batch_all_loss / (i + 1),
                              'Lpri': each_batch_primary_loss / (i + 1),
                              'Lsuper': each_batch_super_loss / (i + 1),
                              'lr': scheduler.get_last_lr()[0]})
            pbar.update(1)
        print("\nFinish Train")
        return each_batch_all_loss / each_data_batch

def val_once(model,t_each_data_batch,epoch,epochs,criterion,optimizer,test_dataloaders):
    model.eval()
    with tqdm(total=t_each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as t_pbar:
        each_batch_psnr = 0
        each_batch_ssim = 0
        print("Start Test")
        with torch.no_grad():
            for i, (img1, mask, lables) in enumerate(test_dataloaders):
                input_img = img1.float().cuda(device=args.device_ids[0])
                mask_img = mask.float().cuda(device=args.device_ids[0])
                lables = lables.float().cuda(device=args.device_ids[0])

                optimizer.zero_grad()
                mesh_final, warp_image_final, warp_mask_final = model.forward(input_img, mask_img)

                I1 = warp_image_final.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                I2 = lables.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

                psnr = compare_psnr(I1, I2 , data_range=1)
                ssim = compare_ssim(I1 , I2, data_range=1, channel_axis=2)

                each_batch_psnr += psnr / args.test_batch_size
                each_batch_ssim += ssim / args.test_batch_size
                t_pbar.set_postfix({'average psnr': each_batch_psnr / (i + 1),
                                    'average ssim': each_batch_ssim / (i + 1)})
                t_pbar.update(1)
        print("\nFinish Test")

def train(model,saveModelName,criterion,optimizer,scheduler,start_epochs=0, end_epochs=1):
    loss_history = []
    each_data_batch = len(dataloders['train'])
    t_each_data_batch = len(dataloders['test'])
    for epoch in range(start_epochs,end_epochs):
        # choice dataloaders
        train_dataloaders = dataloders['train']
        test_dataloaders = dataloders['test']

        # training
        each_batch_all_loss = train_once(model,each_data_batch,epoch, end_epochs,criterion,optimizer,train_dataloaders)
        # testing
        if epoch % 10 == 0:
            val_once(model, t_each_data_batch, epoch, end_epochs, criterion, optimizer,test_dataloaders)

        # learning rate scheduler
        scheduler.step()
        # print("epoch:",epoch,"lr:",scheduler.get_last_lr())
        loss_history.append(each_batch_all_loss)

        if (epoch + 1) % 10 ==0 or epoch >= int(end_epochs-5):
            torch.save(model.state_dict(), saveModelName + "_" + "epoch" + str(epoch + 1) + ".pkl")
            np.save(saveModelName + "_" + "epoch" + str(epoch + 1) + "_" + "TrainLoss" +
            str(round(each_batch_all_loss, 3)), np.array(loss_history))

    show_plot(loss_history)

def show_plot(loss_history):
    counter = range(len(loss_history))
    plt.plot(counter, loss_history)
    plt.legend(['train loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='E:\DataSets\DIR-D')
    parser.add_argument('--dis_mesh_path_TeacherImprove', type=str, default='../DeepRectangling_Teacher2(Improved)/Codes/result/train_final_mesh')
    parser.add_argument('--dis_mesh_path_TeacherOR', type=str, default= '../DeepRectangling_Teacher1(Initial)/train_final_mesh')
    parser.add_argument('--dis_mesh_path_TeacherImprove_weight', type=str, default='../DeepRectangling_Teacher2(Improved)/Codes/result/weight')
    parser.add_argument('--dis_mesh_path_TeacherOR_weight', type=str,  default='../DeepRectangling_Teacher1(Initial)/result/weight')
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--end_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='model/distill_model')
    parser.add_argument('--lam_perception', type=float, default=0.2)
    parser.add_argument('--lam_mask', type=float, default=1)
    parser.add_argument('--lam_mesh', type=float, default=1)
    parser.add_argument('--lam_appearance', type=float, default=1)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_epoch', type=int, default=0)
    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    print(args)
    ##############
    pathGT = os.path.join(args.path, 'training/gt')
    pathInput = os.path.join(args.path, 'training/input')
    pathMask = os.path.join(args.path, 'training/mask')
    pathMesh1 = args.dis_mesh_path_TeacherImprove
    pathMesh2 = args.dis_mesh_path_TeacherOR
    pathMesh1_weight = args.dis_mesh_path_TeacherImprove_weight
    pathMesh2_weight = args.dis_mesh_path_TeacherOR_weight
    pathGT2 = os.path.join(args.path, 'testing/gt')
    pathInput2 = os.path.join(args.path, 'testing/input')
    pathMask2 = os.path.join(args.path, 'testing/mask')
    image_datasets = {}
    image_datasets['train'] = SPRectanglingTrainDataSet2TeachWeight(pathInput, pathMask, pathGT, pathMesh1, pathMesh2, pathMesh1_weight, pathMesh2_weight, args.img_h, args.img_w)
    image_datasets['test'] = SPRectanglingTestDataSet(pathInput2, pathMask2, pathGT2, args.img_h, args.img_w)
    dataloders = {}
    dataloders['train'] = DataLoader(image_datasets['train'], batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dataloders['test'] = DataLoader(image_datasets['test'], batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    # define somethings
    criterion = Distill_loss_2Teacher_Weight(args.lam_appearance, args.lam_perception, args.lam_mask, args.lam_mesh).cuda(device=args.device_ids[0])
    model = RectanglingNetwork()
    # loading model to device 0
    model = model.to(device=args.device_ids[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-4)
    lrScheduler = warmUpLearningRate(args.end_epochs, warm_up_epochs=10, scheduler='cosine')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrScheduler)

    if args.resume:
        for i in range(0, (args.start_epochs + 1)):
            scheduler.step()
        args.start_epochs = args.resume_epoch
        load_path = args.save_model_name + "_" + "epoch" + str(args.resume_epoch) + ".pkl"

        pretrain_model = torch.load(load_path, map_location='cpu')
        # Extract K,V from the existing model
        model_dict = model.state_dict()
        # Create a new weight dictionary and update it
        state_dict = {k: v for k, v in pretrain_model.items() if k in model_dict.keys()}
        # Update the weight dictionary of the existing model
        model_dict.update(state_dict)
        # Load the updated weight dictionary
        model.load_state_dict(model_dict)
        print("-----resume train, load model state dict success------")
    # start train
    train(model, args.save_model_name, criterion, optimizer, scheduler, args.start_epochs, args.end_epochs)

