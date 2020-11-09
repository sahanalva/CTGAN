import numpy as np

import torch
import usample.usample as usample
import emcee

class ConditionalGenerator(object):
    def __init__(self, data, output_info, log_frequency, conditional_cols = None):
        self.model = []
        
        start = 0
        skip = False
        max_interval = 0
        counter = 0
        for item in output_info:
            if item[1] == 'tanh':
                start += item[0]
                skip = True
                continue

            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    start += item[0]
                    continue

                end = start + item[0]
                max_interval = max(max_interval, end - start)
                counter += 1
                self.model.append(np.argmax(data[:, start:end], axis=-1))
                start = end

            else:
                assert 0
        """
        Above loop is taking argmax of the one hot encoding cols (basically columns where 1 is the value) 
        and is storing in self.model. This is done only for the categorical variables
        """
        
        assert start == data.shape[1]

        
        self.conditional_col_index = {}
        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        skip = False
        start = 0
        self.p = np.zeros((counter, max_interval))
        tau = 10.0
        self.class_samples = []
        for item in output_info:
            if item[1] == 'tanh':
                skip = True
                start += item[0]
                continue
            elif item[1] == 'softmax':
                if skip:
                    start += item[0]
                    skip = False
                    continue
                end = start + item[0]
                tmp = np.sum(data[:, start:end], axis=0)
                if log_frequency:
                    tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                self.p[self.n_col, :item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                if conditional_cols != None:
                    if item[2] in conditional_cols:
                        self.conditional_col_index[self.n_col] = (self.n_opt, item[0])
                print("Umbrella Sampling for column:",item[2])
                self.class_samples.append(self.umbrella_sampling(tau, tmp))
                self.n_col += 1
                self.n_opt += item[0]
                
                start = end
            else:
                assert 0

        self.interval = np.asarray(self.interval)

        """
        Above loop calculates probability for categorical variables. We can use log transform if required
        """
    
    def random_choice_prob_index(self, idx):


        val = np.asscalar(np.unique(idx))
        #col_len = np.count_nonzero(self.p[val])
        #col_val = np.random.randint(low=0, high=col_len) * np.ones_like(idx)
        col_val = np.random.choice(self.class_samples[val]) * np.ones_like(idx)
        
        return col_val
    """
    Above function gives the index of category of the corresponding feature that we are setting = 1 
    """

    def sample(self, batch):
        if self.n_col == 0:
            return None

        batch = batch
        idx = np.random.choice(np.arange(self.n_col), batch)    # picking random feature 
        
        if self.conditional_col_index:
            idx = np.random.choice(list(self.conditional_col_index.keys())) * np.ones_like(idx)
        else:
            idx = np.random.randint(low=0, high=self.n_col) * np.ones_like(idx) 

        vec1 = np.zeros((batch, self.n_opt), dtype='float32')   
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        mask1[np.arange(batch), idx] = 1
        opt1prime = self.random_choice_prob_index(idx)          #column index of category chosen wrt to feature index
        opt1 = self.interval[idx, 0] + opt1prime                #column index of catgoery chosen in the condtion matrix.
        vec1[np.arange(batch), opt1] = 1                        #condition matrix

        return vec1, mask1, idx, opt1prime

    """
    Above function returns sampled condtion matrix, mask (one hot encoding of conditioned feature), index of conditioend feature, 
    column index of catgoery chosen in the condtion matrix.
    """

    def sample_zero(self, batch):
        if self.n_col == 0:
            return None

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        mask1[np.arange(batch), idx] = 1
        for i in range(batch):
            col = idx[i]
            pick = int(np.random.choice(self.model[col]))
            vec[i, pick + self.interval[col, 0]] = 1
        return vec,mask1

    
    def gumbel_softmax_fn(self, p, tau,class_prob):

        distribution = torch.distributions.RelaxedOneHotCategorical(torch.tensor(tau), torch.tensor(class_prob))

        return distribution.log_prob(torch.tensor(p))

    def umbrella_sampling(self, tau, class_prob):
        us = usample.UmbrellaSampler(self.gumbel_softmax_fn , lpfargs=[tau,class_prob],   debug=True,  burn_acor=20 )
        temps = np.linspace( 1 , 10 , 8 ) 
        us.add_umbrellas( temps , ic = [0], numwalkers=8 , sampler=emcee.EnsembleSampler )
        pos_us, _, _ = us.run(1000 , freq=1000, repex=10, grstop = 0.01)

        quant_splits = np.linspace(0,1,len(class_prob)+1)[1:]
        quant_splits[:-1] = quant_splits[:-1]+ np.random.uniform(low = -1, high = 1,size = (1,len(quant_splits)-1)) * 0.001
        quant_val = np.quantile(pos_us[:,0],quant_splits)
        sampled_classes = np.digitize(pos_us[:,0], quant_val, right= True)

        return sampled_classes