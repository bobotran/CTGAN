import numpy as np
import torch
from torch import optim
from torch.nn import functional

from ctgan.conditional import ConditionalGenerator
from ctgan.models import Discriminator, Generator
from ctgan.sampler import Sampler
from ctgan.transformer import DataTransformer

from ctgan.glow.models import Glow
import ctgan.glow.util as util


class CTGANSynthesizer(object):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.

    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        gen_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Resiudal Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        dis_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        l2scale (float):
            Wheight Decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
    """

    def __init__(self, args, embedding_dim=128, gen_dim=(256, 256), dis_dim=(256, 256),
                 l2scale=1e-6, batch_size=500):
        self.args = args

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data, discrete_columns=tuple(), epochs=300, log_frequency=True):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a
                pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            epochs (int):
                Number of training epochs. Defaults to 300.
            log_frequency (boolean):
                Whether to use log frequency of categorical levels in conditional
                sampling. Defaults to ``True``.
        """

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dimensions

        assert self.args.batch_size % 2 == 0

        self.flow = Glow(dim=data_dim,
               hidden_layers=self.args.hidden_layers,
               num_levels=self.args.num_levels,
               num_steps=self.args.num_steps).to(self.device)

        loss_fn = util.NLLLoss().to(device)
        optimizer = optim.Adam(self.flow.parameters(), lr=args.lr, betas=(0.5, 0.9),
            weight_decay=self.args.l2scale)

        steps_per_epoch = max(len(train_data) // self.args.batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):
                c1, m1, col, opt = None, None, None, None
                real = data_sampler.sample(self.args.batch_size, col, opt)
                real = torch.from_numpy(real.astype('float32')).to(self.device)

                optimizer.zero_grad()
                z, sldj = self.flow(real, reverse=False)
                loss = loss_fn(z, sldj)
                loss.backward()
                optimizer.step()

            print("Epoch %d, Loss: %.4f" %
                  (i + 1, loss.detach().cpu()),
                  flush=True)

    def sample(self, n):
        """Sample data similar to the training data.

        Args:
            n (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        steps = n // self.args.batch_size + 1
        data = []
        for i in range(steps):
            z = torch.randn((self.args.batch_size, self.transformer.output_dimensions), dtype=torch.float32, device=device)
            fake, _ = self.flow(z, reverse=True)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self.transformer.inverse_transform(data, None)
