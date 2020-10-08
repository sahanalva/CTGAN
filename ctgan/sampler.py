import numpy as np


class Sampler(object):
    """docstring for Sampler."""

    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)
        
        st = 0
        skip = False
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])                  #row index of non zero values in a particular category column of a feature

                self.model.append(tmp)                  #list of lists of row indexes of non zero vals
                st = ed
            else:
                assert 0
        assert st == data.shape[1]

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        #print("sampler,",self.model[0])
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))   #for every data point (n =batch size), we have permuatated col and opt. We use that to find non zero rows in original data for same but shuffled conditions 
        #print("sampler,", idx)
        return self.data[idx]
    """
    Above method returns original data for same but shuffled conditions  
    """