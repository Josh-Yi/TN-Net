import torch

class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.cube_len =42
        self.bias = False
        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1,1,1)

        self.fcs =torch.nn.Sequential(
            torch.nn.BatchNorm1d(208),
            torch.nn.Linear(208,256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(256,512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(512,1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=0.2),

        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1024, self.cube_len*8, kernel_size=4, stride=2, bias=self.bias, padding=padd),
            torch.nn.BatchNorm3d(self.cube_len*8),
            torch.nn.LeakyReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*8, self.cube_len*4, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*4),
            torch.nn.LeakyReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*4, self.cube_len*2, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len*2),
            torch.nn.LeakyReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len*2, self.cube_len, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.BatchNorm3d(self.cube_len),
            torch.nn.LeakyReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, bias=self.bias, padding=(1, 1, 1)),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = x.reshape(-1,208)
        x = self.fcs(x)
        out = x.view(-1, 1024, 1, 1, 1)
        #print(out.size())  # torch.Size([100, 200, 1, 1, 1])
        out = self.layer1(out)
        #print(out.size())  # torch.Size([100, 512, 4, 4, 4])
        out = self.layer2(out)
        #print(out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer3(out)
        #print(out.size())  # torch.Size([100, 128, 16, 16, 16])
        out = self.layer4(out)
        #print(out.size())  # torch.Size([100, 64, 32, 32, 32])
        out = self.layer5(out)
        # print(out.shape)
        out = torch.nn.functional.interpolate(out,scale_factor=tuple([0.5,0.5,0.625]),recompute_scale_factor=True)
        #print(out.size())  # torch.Size([100, 1, 64, 64, 64])
        return out
