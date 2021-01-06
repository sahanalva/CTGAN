import numpy as np
import torch
from torch import optim
from torch.nn import functional
import xgboost as xgb
from conditional import ConditionalGenerator
from models import Discriminator, Generator
from sampler import Sampler
from transformer import DataTransformer
import pickle
import pandas as pd


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

    def __init__(self, embedding_dim=128, gen_dim=(256, 256), dis_dim=(256, 256),
                 l2scale=1e-6, batch_size=500):

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def _cond_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0
        skip = False
        for item in self.transformer.output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True

            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                ed_c = st_c + item[0]
                tmp = functional.cross_entropy(
                    data[:, st:ed],
                    torch.argmax(c[:, st_c:ed_c], dim=1),
                    reduction='none'
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c

            else:
                assert 0

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def fit(self, train_data, prefered_label, black_box_path, discrete_columns=tuple(),conditional_cols = None, epochs=300, log_frequency=True):
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

        self.prefered_label = prefered_label
        self.blackbox_model = pickle.load(open(black_box_path, "rb"))

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dimensions
        self.cond_generator = ConditionalGenerator(
            train_data,
            self.transformer.output_info,
            log_frequency, conditional_cols
        )

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim
        ).to(self.device)



        discriminator = Discriminator(
            data_dim,
            self.dis_dim,1
        ).to(self.device)

        conditonal_discriminator = Discriminator(
            1 + self.cond_generator.n_opt,
            self.dis_dim
        ).to(self.device)

        optimizerG = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9),
            weight_decay=self.l2scale
        )
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        optimizerconditionalD = optim.Adam(conditonal_discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        for i in range(epochs):
            flip_loss_list = []
            real_flip_loss_list = []
            for id_ in range(steps_per_epoch):
                
                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                real = torch.from_numpy(real.astype('float32')).to(self.device)

                if c1 is not None:

                    real_cat = real
                    fake_cat = fakeact

                else:
                    real_cat = real
                    fake_cat = fake

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                if c1 is not None:
                    conditional_fake_cat = torch.cat([y_fake, c1], dim=1)
                    conditional_real_cat = torch.cat([y_real, c2], dim=1)

                else:
                    conditional_fake_cat = y_fake
                    conditional_real_cat = y_real
                
                conditional_y_fake = conditonal_discriminator(conditional_fake_cat)
                conditional_y_real = conditonal_discriminator(conditional_real_cat)
                

                pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, self.device)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                condtional_pen = conditonal_discriminator.calc_gradient_penalty(conditional_real_cat, conditional_fake_cat, self.device)
                loss_condtional_d = -(torch.mean(conditional_y_real) - torch.mean(conditional_y_fake))

                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward(retain_graph=True)
                optimizerD.step()

                optimizerconditionalD.zero_grad()
                condtional_pen.backward(retain_graph=True)
                loss_condtional_d.backward()
                optimizerconditionalD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)
                

                if c1 is not None:
                    y_fake = discriminator(fakeact)
                    conditional_y_fake = conditonal_discriminator(torch.cat([y_fake, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)
                    conditional_y_fake = conditonal_discriminator(y_fake)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                fake_act_inv = self.transformer.inverse_transform(fakeact.detach().cpu().numpy(), None)
                fake_act_inv = _factorize_categoricals(fake_act_inv, discrete_columns)
                fake_act_inv = xgb.DMatrix(data = fake_act_inv)
                black_box_pred_prob = self.blackbox_model.predict(fake_act_inv)
                #black_box_pred_prob = torch.from_numpy(np.stack([1-black_box_pred_prob,black_box_pred_prob], axis = -1))
                #flip_loss = torch.nn.CrossEntropyLoss()(black_box_pred_prob, torch.tensor([self.prefered_label]).repeat(self.batch_size))
                flip_loss = sum(-np.log(black_box_pred_prob))/self.batch_size


                real_inv = self.transformer.inverse_transform(real.detach().cpu().numpy(), None)
                real_inv = _factorize_categoricals(real_inv, discrete_columns)
                real_inv = xgb.DMatrix(data = real_inv)
                real_pred_prob = self.blackbox_model.predict(real_inv)
                #black_box_pred_prob = torch.from_numpy(np.stack([1-black_box_pred_prob,black_box_pred_prob], axis = -1))
                #flip_loss = torch.nn.CrossEntropyLoss()(black_box_pred_prob, torch.tensor([self.prefered_label]).repeat(self.batch_size))
                real_flip_loss = sum(-np.log(real_pred_prob))/self.batch_size

                
                loss_g = -torch.mean(conditional_y_fake) +  cross_entropy + 10*flip_loss
                #print(f"Base Loss:{-torch.mean(conditional_y_fake)}, Conditional Loss:{cross_entropy}, 10Flip Loss:{flip_loss}")
                flip_loss_list.append(flip_loss)
                real_flip_loss_list.append(real_flip_loss)

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            print(f"Generated flip loss {np.mean(flip_loss_list)}, Real flip loss {np.mean(real_flip_loss_list)}")
            print("Condtional Cross Entropy Loss", cross_entropy)
            print("Epoch %d, Loss G: %.4f, Loss D: %.4f, Loss Conditional D: %.4f" %
                  (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu(), loss_condtional_d.detach().cpu()),
                  flush=True)
            

    def sample(self, n, col_index = None):
        """Sample data similar to the training data.
        Args:
            n (int):
                Number of rows to sample.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec,m1 = self.cond_generator.sample_zero(self.batch_size)
            m1 = torch.from_numpy(m1).to(self.device)
            if condvec is None:
                pass
            else:
                c1 = condvec
                if col_index != None:
                    c1 = np.zeros_like(c1)
                    c1[:,col_index] = 1
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)
                

            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())
            print(self._cond_loss(fake, c1, m1))

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self.transformer.inverse_transform(data, None)
    
def _factorize_categoricals(df, discrete_columns):
    for col in discrete_columns:
        df[col], _ = pd.factorize(df[col])
    return df 
