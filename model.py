# layers

class SHT(nn.Module):
    def __init__(self, L, Y_inv, area):
        """
        Input shape  : [batch, n_ch, 120]
        Output shape : [batch, n_ch, (8+1)**2]
        """

        super().__init__()

        self.Y_inv = Y_inv[:, : (L + 1) ** 2]
        self.area = area

    def forward(self, x):
        x = torch.mul(self.area, x)
        x = torch.matmul(x, self.Y_inv)

        return x

class SHConv(nn.Module):
    def __init__(self, in_ch, out_ch, L):
        """
        Input shape  : [batch, in_ch, (L+1)**2]
        Output shape : [batch, out_ch, (L+1)**2]
        """

        super().__init__()

        ncpt = L + 1

        self.weight = nn.Parameter(torch.empty(in_ch, out_ch, ncpt))
        self.repeats = nn.Parameter(torch.tensor([(2 * l + 1) for l in range(L + 1)]), requires_grad=False)

        stdv = 1.0 / math.sqrt(in_ch * (L + 1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):

        w = torch.repeat_interleave(self.weight, self.repeats, dim=2)
        x = torch.mul(w.unsqueeze(0), x.unsqueeze(2)).sum(1)

        return x


class ISHT1(nn.Module):
    def __init__(self, Y):
        """
        Input shape  : [batch, n_ch, (L+1)**2]
        Output shape : [batch, n_ch, 120]
        """

        super().__init__()

        self.Y = Y

    def forward(self, x):
        x = torch.matmul(x, self.Y[: x.shape[-1], :])

        return x

class ISHT2(nn.Module):
    def __init__(self, Y480_289):
        """
        Input shape  : [batch, n_ch, (L+1)**2]
        Output shape : [batch, n_ch, 480]
        """

        super().__init__()

        self.Y480_289 = Y480_289

    def forward(self, x):
        x = torch.matmul(x, self.Y480_289)

        return x


# model

class SCNN(nn.Module):
    def __init__(self, Y, Y_inv, area,Y480_289, in_ch, out_ch, L,nonlinear=None, fullband=True, bn=True):
        """
        In channel shape  : [batch, in_ch, n_vertex]
        Out channel shape : [batch, out_ch, n_vertex]
        """
        super().__init__()

        self.first = nn.Sequential(SHT(8, Y_inv, area),ISHT1(Y))
        self.shconv1 = nn.Sequential(SHT(L, Y_inv, area), SHConv(in_ch, 93*2, L), ISHT1(Y))
        self.shconv2 = nn.Sequential(SHT(16, Y_inv, area), SHConv(93*2, out_ch, 16), ISHT1(Y))
        self.final = nn.Sequential(SHT(16, Y_inv, area),ISHT2(Y480_289))

        self.impulse1 = nn.Conv1d(in_ch, 93*2, kernel_size=1, stride=1, bias=not bn) if fullband else lambda _: 0
        self.impulse2 = nn.Conv1d(93*2, in_ch, kernel_size=1, stride=1, bias=not bn) if fullband else lambda _: 0
        self.nonlinear = F.relu if nonlinear is not None else nn.Identity()

    def forward(self, x):
        x = self.first(x)

        x = self.shconv1(x) + self.impulse1(x)
        x = self.nonlinear(x)

        x = self.shconv2(x) + self.impulse2(x)
        x = self.nonlinear(x)

        x = self.final(x)

        return x
