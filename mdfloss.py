import torch
import torch.nn as nn


class MDFLoss(nn.Module):
    def __init__(self, saved_ds_path, cuda_available = True):
        super(MDFLoss, self).__init__()

        if cuda_available:
            self.Ds = torch.load(saved_ds_path)
        else:
            self.Ds = torch.load(saved_ds_path, map_location=torch.device('cpu'))

        self.num_discs = len(self.Ds)

    def forward(self, x, y, num_scales=8, is_ascending=1):
        # Get batch_size
        batch_size = x.shape[0]
        
        # Initialize loss vector
        loss = torch.zeros([batch_size]).to(x.device)
        # For every scale
        for scale_idx in range(num_scales):
            # Reverse if required
            if is_ascending:
                scale = scale_idx
            else:
                scale = self.num_discs - 1 - scale_idx

            # Choose discriminator
            D = self.Ds[scale]
            
            # Get discriminator activations
            pxs = D(x, is_loss=True)
            pys = D(y, is_loss=True)

            # For every layer in the output
            for idx in range(len(pxs)):
                # Compute L2 between representations
                l2 = (pxs[idx] - pys[idx])**2
                l2 = torch.mean(l2, dim=(1, 2, 3))

                # Add current difference to the loss
                loss += l2

        # Mean loss
        loss = torch.mean(loss)

        return loss