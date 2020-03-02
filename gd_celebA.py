import torch
from torch import nn
from os.path import isfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from skimage.measure import compare_ssim as ssim

train_dataset='CelebA'
test_dataset='CelebA'
z_dim=256
sub_dim_pool=[5000]#
sub_dim_mse=[]
sub_dim_psnr=[]
sub_dim_ssim=[]
sub_dim_rec=[]
for sub_dim in sub_dim_pool:
    print(sub_dim)
#    sub_dim=128#512
    subtype_pool=['abs']# 'abs','linear','square','original'
    train_z_type='PCA'
    test_z_type='Gauss'#'Gauss'
    train_epochs=500
    
    test_epochs=1500
    test_outer_epochs=50
    
    train_batch_size=256
    test_batch_size=10
    test_size=10 # choose something greater than or equal to test_batcch_size (otherwise test_alpha should be normalized)

    opt='SGD'
    if opt=='SGD':
        train_lr=1
    elif opt=='Adam':
        train_lr=1e-3
    train_alpha=200

    z_norm_type='unit_norm'
    
    x_init='random'#'power_project','spectral'
    seed=100
    random_restart=1

  
    
    train=0
    test=1
    mask_id=10

    rotation_theta=0
    start_fig=0
    mask_ratio=0.9
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    nz = z_dim
    ngf = 64
    ndf =64
    if train_dataset=='MNIST' or train_dataset=='SVHN_gray' or train_dataset=='EMNIST_digit' or train_dataset=='FashionMNIST':
        nc = 1
    elif train_dataset=='SVHN' or train_dataset=='CIFAR10' or train_dataset=='CelebA':
        nc = 3
        
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
    #            nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2,1, bias=False),
    #            nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
    #            nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
    #            nn.BatchNorm2d(ngf),
                nn.ReLU(True),
    ##            # state size. (ngf) x 32 x 32
    #            nn.ConvTranspose2d(    ngf,      ngf/2, 4, 2, 1, bias=False),
    #            #nn.BatchNorm2d(ngf/2),
    #            nn.ReLU(True),
    #            # state size. (ngf/2) x 64x 64
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 128 x 128
            )
    
        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            return output
        
    loss_subtype=[]  
    loss_outer_subtype=[]
    for subtype in subtype_pool: 
    
    
        test_alpha=3.84e-5*np.float(train_alpha)*test_batch_size/train_batch_size#200
            
        datax=np.load('/media/rakib/Data/Data/GLO/Feature/celebA_test2_resized_'+str(ngf)+'_'+str(ndf)+'.npz')
        x_test=datax['x_test']
        x_test_org=x_test
        if rotation_theta!=0:
            x_rot=np.zeros((x_test.shape[0],nc,ngf,ndf))
            for i in range (0, x_test.shape[0]):
                for j in range (0, nc):
                    x_rot[i,j,:,:]=rotate(x_test[i,j,:,:],rotation_theta,reshape=False)
            x_test=x_rot       

        #    dataz=np.load('/media/rakib/Data/Data/GLO/Feature/'+test_dataset+'_'+test_z_type+'_'+str(z_dim)+'.npz')
        #    z_test=dataz['z_init_test']
        if test_z_type=='Gauss':
            z_test=np.zeros((x_test.shape[0],z_dim))
            for i in range (0,z_test.shape[0]):
                np.random.seed(seed)
                z_test[i,:]=np.random.normal(loc=0.0, scale=1.0, size=(1,z_dim))
        
        elif test_z_type=='Optimum':
            z_test=np.load( '/media/rakib/Data/Data/GLO/Results/'+train_dataset+'_'+'Gauss'+'_'+str(z_dim)+'_alpha'+str(np.float(train_alpha))+'_lr'+str(train_lr)+'_epochs'+str(200)+'wobatchnorm.npy') 
        ## Select test size
        if test_size !=-1:
            x_test=x_test[0:test_size,:,:,:]            
            x_test_org=x_test_org[0:test_size,:,:,:]
            z_test=z_test[0:test_size,:]
    
    
    #    generator = torch.load(  '/media/rakib/Data/Data/GLO/Model/'+train_dataset+'_'+train_z_type+'_'+str(z_dim)+'_alpha'+str(train_alpha)+'_lr'+str(train_lr)+'_epochs'+str(train_epochs)+'wobatchnorm')
    #    optimizer = torch.optim.SGD(generator.parameters(), train_lr)
    
        # Random Matrix
            # Random Matrix
        if subtype=='linear' or subtype=='abs' or subtype=='square':
            if isfile('/media/rakib/Data/Data/GLO/Feature/random_mask_'+str(test_batch_size)+'_'+str(sub_dim)+'_'+str(nc*ngf*ndf)+'_'+str(mask_id)+'.npy'):
                mask=np.load('/media/rakib/Data/Data/GLO/Feature/random_mask_'+str(test_batch_size)+'_'+str(sub_dim)+'_'+str(nc*ngf*ndf)+'_'+str(mask_id)+'.npy')
            else:
                if mask_id>0:
                    mask=np.random.normal(loc=0.0, scale=1.0/np.sqrt(sub_dim), size=(1,sub_dim,nc*ngf*ndf))
                else:
                    mask=(np.random.randn(test_batch_size,sub_dim,nc*ngf*ndf)+1j*np.random.randn(test_batch_size,sub_dim,nc*ngf*ndf))/np.sqrt(2*sub_dim)
                np.save('/media/rakib/Data/Data/GLO/Feature/random_mask_'+str(test_batch_size)+'_'+str(sub_dim)+'_'+str(nc*ngf*ndf)+'_'+str(mask_id),mask)
#            for i in range (0,test_batch_size):
    #            mask[i]=mask[i]/np.linalg.norm(mask[i],ord=2)
#                mask[i]=mask[i]/np.sqrt(sub_dim)
            mask_tensor=torch.cuda.FloatTensor(mask).view(-1,sub_dim,nc*ngf*ndf)
            
    
    # Applying mask
        if subtype=='linear':
            x_test=2*x_test-1
            x_test_temp=np.zeros((x_test.shape[0],sub_dim,1))
            for i in range (0, x_test.shape[0]):
                x_test_temp[i,:,:]=np.matmul(mask[0,:,:],x_test[i,:,:,:].flatten().reshape(1,nc*ngf*ndf,1))
            x_test=x_test_temp
        elif subtype=='abs': 
            x_test=2*x_test-1
            x_test_temp=np.zeros((x_test.shape[0],sub_dim,1))
            for i in range (0, x_test.shape[0]):
                x_test_temp[i,:,:]=np.abs(np.matmul(mask[0,:,:],x_test[i,:,:,:].flatten().reshape(1,nc*ngf*ndf,1)))
            x_test=x_test_temp
        elif subtype=='square':
            x_test=2*x_test-1
            x_test_temp=np.zeros((x_test.shape[0],sub_dim,1))
            for i in range (0, x_test.shape[0]):
                x_test_temp[i,:,:]=np.square(np.matmul(mask[0,:,:],x_test[i,:,:,:].flatten().reshape(1,nc*ngf*ndf,1)))
            x_test=x_test_temp
    
    
        for i in range(z_test.shape[0]):
            z_test[i] = z_test[i, :] / np.linalg.norm(z_test[i, :], 2)
    

        test_size=x_test.shape[0]
        batch_no=np.int(np.ceil(test_size/test_batch_size))
        idx=np.arange(test_size)
        
        x_rec=np.zeros((x_test.shape[0],nc,ngf,ndf))
        x_rec1=np.zeros((x_test.shape[0],nc,ngf,ndf))
        

        loss_test=[]
        for batch_idx in range(0,batch_no):
            x_best=np.zeros((test_batch_size,nc*ngf*ndf,1))
#            z_test_temp=z_test
            for rr in range (0,random_restart):
                generator = torch.load(  '/media/rakib/Data/Data/GLO/Model/'+train_dataset+'_'+train_z_type+'_'+str(z_dim)+'_alpha'+str(train_alpha)+'_lr'+str(train_lr)+'_epochs'+str(train_epochs)+'wobatchnorm')
                if opt=='SGD':
                    optimizer = torch.optim.SGD(generator.parameters(), train_lr)
                elif opt=='Adam':
                    optimizer = torch.optim.Adam(generator.parameters(), train_lr) 
        #        if batch_idx%100==0:
    #            print(batch_idx)
                epoch_idx=idx    
                y=x_test[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,0].reshape(test_batch_size,sub_dim,1)
                x_true=2*x_test_org[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:,:,:].reshape(test_batch_size,nc*ngf*ndf,1)-1
                

                ## Beginning
                

                    
                loss_epoch=[]
    
                    
#                    z_batch=z_test[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:]
#                    np.random.seed(seed)
#                    z_batch= np.random.normal(loc=0.0, scale=1.0, size=(test_batch_size,z_dim))
#                    for i in range(z_batch.shape[0]):
#                        z_batch[i] = z_batch[i, :] / np.linalg.norm(z_batch[i, :], 2)
                        
                z_batch=z_test[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:]
#                    np.random.seed(seed)
#                z_batch= np.random.normal(loc=0.0, scale=1.0, size=(test_batch_size,z_dim))
#                for i in range(z_batch.shape[0]):
#                    z_batch[i] = z_batch[i, :] / np.linalg.norm(z_batch[i, :], 2)
                    
                for epoch in range (0,test_epochs):
                    if subtype=='linear' or subtype=='abs' or subtype=='square':
                        x_batch_tensor=torch.cuda.FloatTensor(y).view(-1, y.shape[1],1)
    
                    z_batch_tensor=torch.autograd.Variable(torch.cuda.FloatTensor(z_batch).view(-1, z_dim, 1, 1),requires_grad=True)
    
                    x_hat = generator(z_batch_tensor)
                    if subtype=='linear':
                        x_hat_mask=torch.matmul(mask_tensor,x_hat.view(-1, nc* ngf*ndf,1))
                    elif subtype=='abs':
                        x_hat_mask=torch.abs(torch.matmul(mask_tensor,x_hat.view(-1, nc* ngf*ndf,1)))
                    elif subtype=='square':
                        x_hat_mask=torch.matmul(mask_tensor,x_hat.view(-1, nc* ngf*ndf,1))
                        x_hat_mask=torch.mul(x_hat_mask,x_hat_mask)
                    elif subtype=='original':
                        x_hat_mask=x_hat
    
                    loss=sub_dim*(x_hat_mask - x_batch_tensor).pow(2).mean()
                    loss_epoch.append(loss.item())
    
                    optimizer.zero_grad()
                    loss.backward()        
        #            optimizer.step()
    
                    with torch.no_grad():        
                        z_grad = z_batch_tensor.grad.data.cuda()    
                        z_update = z_batch_tensor - test_alpha * z_grad
                        z_update = z_update.cpu().numpy()
                        z_update=np.reshape(z_update,z_batch.shape)
                        if z_norm_type=='unit_norm':
                            for i in range(z_update.shape[0]):
                                z_update[i,:] = z_update[i, :] / np.linalg.norm(z_update[i, :], 2)
                        z_batch=z_update    
                        z_test[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:]=z_update
                        
                z_update_tensor=torch.autograd.Variable(torch.cuda.FloatTensor(z_update).view(-1, z_dim, 1, 1))
                x_hat = generator(z_update_tensor)
                x=np.reshape(x_hat.cpu().detach().numpy(),x_true.shape)
                
                if np.mean((x_true-x)**2)<np.mean((x_true-x_best)**2) or random_restart==1:
                    x_best=x
                    x_rec[epoch_idx[batch_idx*test_batch_size:np.min([(batch_idx+1)*test_batch_size,test_size])],:]=x.reshape(x.shape[0],nc,ngf,ndf)
                    loss_epoch_best=loss_epoch

            #post process
            x_rec_org=x_rec
            for i in range (0,x_rec.shape[0]):
#                print(i)
                x_pos=np.abs(np.matmul(mask[0,:,:],x_rec[i,:,:,:].flatten().reshape(1,nc*ngf*ndf,1)))
                x_neg=np.abs(np.matmul(mask[0,:,:],-x_rec[i,:,:,:].flatten().reshape(1,nc*ngf*ndf,1)))
                x_t=x_test[i]
                if np.mean((x_pos-x_t)**2)>np.mean((x_neg-x_t)**2):
#                    print(i)
                    x_rec_org[i]=-x_rec[i]
            temp=x_rec
            x_rec=x_rec_org
            x_rec_org=temp
#                z_test=z_test_temp
                    #loss_test .append(np.array(loss_epoch))
        #            plt.figure()
        #            plt.plot( np.array(loss_epoch))
    #            plt.plot(np.array(loss_outer_epoch))
    #            plt.show()
            loss_test .append(np.array(loss_epoch_best))

            
    #            plt.figure()
    #            plt.plot(loss_outer_epoch)
    #            plt.title('x update loss during testing')
    #            plt.show()
    
        np.savez('/media/rakib/Data/Data/GLO/icassp2019/Results/'+train_dataset+'_'+test_z_type+'_'+str(z_dim)+z_norm_type+'_alpha'+str(test_alpha)+'_epochs'+str(test_epochs)+'_'+str(test_batch_size)+'phase_retrieval_direct_'+subtype+'_subsample_'+str(sub_dim),z_test=z_test,mask=mask,x_rec=x_rec,x_rec_org=x_rec_org)
    

        loss_subtype.append(np.array(loss_test))


    
        #    x_test=x_test/2+0.5
        x_rec=x_rec/2+0.5
        x_rec1=x_rec1/2+0.5
        mse=np.mean((x_rec-x_test_org)**2)
        print(mse)
        psnr=20*np.log10((np.max(x_test_org)-np.min(x_test_org))/np.sqrt(mse))
        

        print(psnr)
        
        ssim_mnist=np.zeros(x_test_org.shape[0])
        for i in range (0,x_test_org.shape[0]):
            img1=x_test_org[i]
            img2=x_rec[i]
            if nc==3:
                img_true=np.zeros((img1.shape[1],img1.shape[2],img1.shape[0]))
                img_rec=np.zeros((img2.shape[1],img2.shape[2],img2.shape[0]))
                for chan in range (0,nc):
                    img_true[:,:,chan]=img1[chan,:,:]
                    img_rec[:,:,chan]=img2[chan,:,:]
                ssim_mnist[i]=ssim(img_true, img_rec,data_range=img_rec.max() - img_rec.min(), multichannel=True)
            elif nc==1:
                img_true=img1.reshape(ngf,ndf)  
                img_rec=img2.reshape(ngf,ndf)  
                ssim_mnist[i]=ssim(img_true, img_rec,data_range=img_rec.max() - img_rec.min())
        
        print(np.mean(ssim_mnist))
        sub_dim_ssim.append(np.mean(ssim_mnist))
        sub_dim_mse.append(mse)
        sub_dim_psnr.append(psnr)
        sub_dim_rec.append(x_rec)
#        print(subtype)
#        print(test_alpha)
        
        x_diff=x_test_org-x_rec
        
        mm=np.mean((x_test_org-x_rec)**2,axis=(1,2,3)) 
        mmx=np.argsort(mm)
        midx=np.int(len(mmx)/2)
        #figset=[mmx[0+start_fig],mmx[1+start_fig],mmx[2+start_fig],mmx[midx-1-start_fig],mmx[midx],mmx[midx+1+start_fig],mmx[-3-start_fig],mmx[-2-start_fig],mmx[-1-start_fig]]
        figset=np.arange(10)

        print(figset)
        plt.figure(figsize=(20, 6))
        n=10
        for i in range(n):
            # display original
            if nc==1:
                ax = plt.subplot(3, n, i + 1)
                plt.imshow(x_rec[figset[i]].reshape(ngf, ndf))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    
                # display reconstruction difference
                ax = plt.subplot(3, n, i + 1 +n)
                plt.imshow(x_rec1[figset[i]].reshape(ngf, ndf))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    
                ax = plt.subplot(3, n, i + 1+2*n)
                plt.imshow(x_test_org[figset[i]].reshape(ngf, ndf))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    
    
            elif nc==3:
                ax = plt.subplot(3, n, i + 1)
                temp=x_rec[figset[i]]
                temp1=np.zeros((ngf, ndf,nc))
                for chan in range (0,nc):
                    temp1[:,:,chan]=temp[chan,:,:]
                plt.imshow(temp1)
    
    #            plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    
                # display reconstruction difference
                ax = plt.subplot(3, n, i + 1 +n)
                temp=x_rec1[figset[i]]
                temp1=np.zeros((ngf, ndf,nc))
                for chan in range (0,nc):
                    temp1[:,:,chan]=temp[chan,:,:]
                plt.imshow(temp1)
    #            plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    
                ax = plt.subplot(3, n, i + 1+2*n)
                temp=x_test_org[figset[i]]
                temp1=np.zeros((ngf, ndf,nc))
                for chan in range (0,nc):
                    temp1[:,:,chan]=temp[chan,:,:]
                plt.imshow(temp1)
    #            plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        #    plt.savefig(test_dataset+'_test_rec_test_epoch_'+str(test_epochs))  
        plt.show()
        


    
    for i in range (0,len(subtype_pool)):
        plt.figure()
        plt.plot(np.mean(loss_subtype[i],axis=0))
        plt.xlabel('Epochs')
        plt.ylabel('Reconstruction loss (Projection step) during testing')
        plt.title('Reconstruction loss (Projection step) vs epochs during testing')

    plt.show()

	

plt.figure()
plt.plot(np.array(sub_dim_pool),np.array(sub_dim_mse),'*-')
plt.xlabel('Number of measurements (m)')
plt.ylabel('Reconstruction error (per pixel)')
plt.show()

plt.figure()
plt.plot(np.array(sub_dim_pool),np.array(sub_dim_psnr),'*-')
plt.xlabel('Number of measurements (m)')
plt.ylabel('PSNR')
plt.show()

plt.figure()
plt.plot(np.array(sub_dim_pool),np.array(sub_dim_ssim),'*-')
plt.xlabel('Number of measurements (m)')
plt.ylabel('Mean SSIM')
plt.show()

fig_row=len(sub_dim_pool)+1
plt.figure(figsize=(20, fig_row*2))
n=10
for i in range(n):

    if nc==1:
        for sublen in range(0,len(sub_dim_pool)):
            
            ax = plt.subplot(fig_row, n, i + 1+sublen*n)
            tempfig=sub_dim_rec[sublen]
            plt.imshow(tempfig[i].reshape(ngf, ndf))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(str(sub_dim_pool[sublen ]))
            
        ax = plt.subplot(fig_row, n, i + 1+fig_row*n-n)
        plt.imshow(x_test_org[figset[i]].reshape(ngf, ndf))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Original')

    elif nc==3:
        for sublen in range(0,len(sub_dim_pool)):
            ax =plt.subplot(fig_row, n, i + 1+sublen*n)
            tempfig=sub_dim_rec[sublen]
            temp=tempfig[figset[i]]
            temp1=np.zeros((ngf, ndf,nc))
            for chan in range (0,nc):
                temp1[:,:,chan]=temp[chan,:,:]
            plt.imshow(temp1)
#            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(str(sub_dim_pool[sublen ]))

        ax = plt.subplot(fig_row, n, i + 1+fig_row*n-n)     
        temp=x_test_org[figset[i]]
        temp1=np.zeros((ngf, ndf,nc))
        for chan in range (0,nc):
            temp1[:,:,chan]=temp[chan,:,:]
        plt.imshow(temp1)
#            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Original')
#    plt.savefig(test_dataset+'_test_rec_test_epoch_'+str(test_epochs))  
plt.show()
torch.cuda.empty_cache()
