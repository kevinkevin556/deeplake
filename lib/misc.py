from torch import nn


class Concat(nn.Module):
    def __init__(self, *nets):
        super().__init__()
        self.nets = nets
        self.__class__.__name__ = nets[0].__class__.__name__
        self.__class__.__qualname__ = nets[0].__class__.__qualname__

    def forward(self, *x):
        out = self.nets[0](*x)
        for m in self.nets[1:]:
            out = m(*out) if isinstance(out, (list, tuple)) else m(out)
        return out

    def train(self, mode=True):
        for m in self.nets:
            m.train(mode)
