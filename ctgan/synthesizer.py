import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.nn import functional

from ctgan.conditional import ConditionalGenerator
from ctgan.models import Discriminator, Generator
from ctgan.sampler import Sampler
from ctgan.transformer import DataTransformer

from ctgan.glow.models import Glow
import ctgan.glow.util as util
import time
from sdgym.synthesizers import BaseSynthesizer
from imblearn.over_sampling import SMOTE

import os
import logging
import json

class CTGlowSynthesizer(BaseSynthesizer):
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

        #os.makedirs(self.args.output_dir)
        with open(os.path.join(self.args.output_dir, 'train_config.json'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=4)
        
        self.label_column_idx = None
        self.transformed_data = None
        self.labels = None

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item[1] == 'softmax':
                ed = st + item[0]
                data_t.append(functional.gumbel_softmax(data[:, st:ed], tau=0.2))
                st = ed
            else:
                assert 0

        return torch.cat(data_t, dim=1)

    def fit(self, train_data, meta=None, discrete_columns=tuple(), ordinal_columns=tuple()):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a
                pandas.DataFrame.
            discrete_columns/ordinal_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        label_column_idx = [i for i, col in enumerate(meta['columns']) if col['name']=='label']
        self.label_column_idx = label_column_idx[0] if len(label_column_idx) > 0 else None
        if self.args.output_latent:
            np.save(os.path.join(self.args.output_dir, 'ctglow_data.npy'), train_data)
        self.labels = train_data[:,self.label_column_idx]
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)

        self.data_dim = self.transformer.output_dimensions
        if self.data_dim % 2 != 0:
            train_data = np.concatenate((train_data, np.zeros((train_data.shape[0], 1))), axis=1)
            self.data_dim += 1

        self.transformed_data = train_data
        if self.args.output_latent:
            np.save(os.path.join(self.args.output_dir, 'alpha_beta_data.npy'), self.transformed_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)
        self.cond_generator = ConditionalGenerator(train_data, self.transformer.output_info, True)

        assert self.args.batch_size % 2 == 0

        self.flow = Glow(dim=self.data_dim,
               hidden_layers=self.args.hidden_layers,
               num_levels=self.args.num_levels,
               num_steps=self.args.num_steps).to(self.device)

        with open(os.path.join(self.args.output_dir, 'architecture.txt'), 'w') as f:
            f.write(self.flow.__str__())

        loss_fn = util.NLLLoss().to(self.device)
        optimizer = optim.Adam(self.flow.parameters(), lr=self.args.lr, betas=(0.5, 0.9),
            weight_decay=self.args.l2scale)
        scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: self.args.lr_decay)

        steps_per_epoch = max(len(train_data) // self.args.batch_size, 1)
        for i in range(self.args.epochs):
            start = time.time()
            for id_ in range(steps_per_epoch):
                condvec = self.cond_generator.sample(self.args.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec

                real = data_sampler.sample(self.args.batch_size, col, opt)
                real = torch.from_numpy(real.astype('float32')).to(self.device)

                z, sldj = self.flow(real, reverse=False)
                loss = loss_fn(z, sldj)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            end = time.time()
            scheduler.step()
            print("Epoch %d, Loss: %.4f, lr: %.8f Time: %.4f" %
                  (i + 1, loss.detach().cpu(), optimizer.param_groups[0]['lr'], end - start))

    @torch.no_grad()
    def sample(self, n):
        """Sample data similar to the training data.

        Args:
            n (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        if self.args.smote == 'augment':
            steps = len(self.transformed_data) // self.args.batch_size + 1
            latent = []
            for i in range(steps):
                start_idx = i * self.args.batch_size
                end_idx = min((i+1) * self.args.batch_size, len(self.transformed_data))

                batch = self.transformed_data[start_idx:end_idx]
                batch = torch.from_numpy(batch.astype('float32')).to(self.device)
                z, _ = self.flow(batch, reverse=False)
                latent.append(z.detach().cpu().numpy())
    
            latent = np.concatenate(latent, axis=0)
            if self.args.output_latent:
                np.save(os.path.join(self.args.output_dir, 'ctglow_latent.npy'), latent)
            sm = SMOTE(random_state=self.args.seed)
            latent_new, _ = sm.fit_resample(latent, self.labels)
   
            steps = len(latent_new) // self.args.batch_size + 1
            data = [] 
            for i in range(steps):
                start_idx = i * self.args.batch_size
                end_idx = min((i+1) * self.args.batch_size, len(latent_new))

                batch = latent_new[start_idx:end_idx]
                batch = torch.from_numpy(batch.astype('float32')).to(self.device)
                smoted, _ = self.flow(batch, reverse=True)
                smotedact = self._apply_activate(smoted)
                data.append(smotedact.detach().cpu().numpy())

            data = np.concatenate(data, axis=0)
            np.random.shuffle(data)
            data = data[:n]

        elif self.args.smote == 'latent':
            steps = n // self.args.batch_size + 1
            latent, labels = [], []
            for i in range(steps):
                z = torch.randn((self.args.batch_size, self.data_dim), dtype=torch.float32, device=self.device)
                latent.append(z.detach().cpu().numpy())

                fake, _ = self.flow(z, reverse=True)
                fakeact = self._apply_activate(fake)
                y = self.transformer.inverse_transform(fakeact.detach().cpu().numpy(), None)[:,self.label_column_idx]
                labels.append(y)

            latent = np.concatenate(latent, axis=0)
            labels = np.concatenate(labels, axis=0)
            sm = SMOTE(random_state=self.args.seed)
            latent_new, _ = sm.fit_resample(latent, labels)

            np.random.shuffle(latent_new)
            latent_new = torch.tensor(latent_new, device=self.device)

            data = [] 
            for i in range(steps):
                fake, _ = self.flow(latent_new[i*self.args.batch_size:(i+1)*self.args.batch_size], reverse=True)
                fakeact = self._apply_activate(fake)
                data.append(fakeact.detach().cpu().numpy())

            data = np.concatenate(data, axis=0)
            data = data[:n]

        elif self.args.smote == 'alpha_beta':
            steps = n // self.args.batch_size + 1
            data, labels = [], []
            for i in range(steps):
                z = torch.randn((self.args.batch_size, self.data_dim), dtype=torch.float32, device=self.device)

                fake, _ = self.flow(z, reverse=True)
                data.append(fake.detach().cpu().numpy())
                fakeact = self._apply_activate(fake)
                y = self.transformer.inverse_transform(fakeact.detach().cpu().numpy(), None)[:,self.label_column_idx]
                labels.append(y)

            data = np.concatenate(data, axis=0)
            labels = np.concatenate(labels, axis=0)
            sm = SMOTE(random_state=self.args.seed)
            data_new, _ = sm.fit_resample(data, labels)

            np.random.shuffle(data_new)
            data_new = torch.tensor(data_new, device=self.device)

            data = [] 
            for i in range(steps):
                fake, _ = self.flow(data_new[i*self.args.batch_size:(i+1)*self.args.batch_size], reverse=True)
                fakeact = self._apply_activate(fake)
                data.append(fakeact.detach().cpu().numpy())

            data = np.concatenate(data, axis=0)
            data = data[:n]

        else:
            steps = n // self.args.batch_size + 1
            data = []
            for i in range(steps):
                z = torch.randn((self.args.batch_size, self.data_dim), dtype=torch.float32, device=self.device)
                fake, _ = self.flow(z, reverse=True)
                fakeact = self._apply_activate(fake)
                data.append(fakeact.detach().cpu().numpy())
    
            data = np.concatenate(data, axis=0)
            data = data[:n] 
 
        return self.transformer.inverse_transform(data, None)
