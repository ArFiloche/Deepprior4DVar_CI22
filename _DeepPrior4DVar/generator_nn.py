import torch.nn as nn

# custom weights initialization called on netG
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

      
class Generator(nn.Module):
    def __init__(self,nc=3,ngf=64,nz=100):
        super(Generator, self).__init__()
        
        # Number of channels in the training images. (3 for eta,u,v)
        self.nc=nc
        # Size of feature maps in generator
        self.ngf=ngf
        # Size of z latent vector (i.e. size of generator input)
        self.nz=nz
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.nz, self.ngf * 8, 4, 1, 0, bias=False),

            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.Upsample(scale_factor = 2, mode='bilinear',align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.ngf * 8, self.ngf * 4,kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.Upsample(scale_factor = 2, mode='bilinear',align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.ngf * 4, self.ngf * 2,kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.Upsample(scale_factor = 2, mode='bilinear',align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.ngf * 2, self.ngf * 1,kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.Upsample(scale_factor = 2, mode='bilinear',align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.ngf, self.nc ,kernel_size=3, stride=1, padding=0),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)