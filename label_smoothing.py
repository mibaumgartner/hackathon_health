from torch import nn


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        logprobs = self.log_softmax(x)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
