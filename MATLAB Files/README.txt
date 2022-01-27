MATLAB Data-sets:

*/ Quick-start set up for the MATLAB data sets including normalisation when needed /*


** elevators.mat **
--------------------

data = loadmat(PATH)['data']
np.random.shuffle(data)
data = torch.Tensor(data)
X = data[:, :-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:, -1]

--------------------
--------------------

** bike.mat **
--------------------

data = loadmat(PATH)['data']
np.random.shuffle(data)
data = torch.Tensor(data)
X = data[:, :-1]
X = X - X.min(0)[0]
X = 2.0 * (X / X.max(0)[0]) - 1.0
y = data[:, -1]
y -= y.mean()
y /= y.std()

--------------------
--------------------

** protein.mat **
--------------------

data = loadmat(PATH)['data']
np.random.shuffle(data)
data = torch.Tensor(data)
N = data.shape[0]
# make train/val/test
n_train = int(0.8 * N)
train_x, train_y = data[:n_train, :-1], data[:n_train, -1]
test_x, test_y = data[n_train:, :-1], data[n_train:, -1]

-------------------
-------------------

** 3droad.mat **
-------------------

data = loadmat(PATH)['data']
np.random.shuffle(data)
data = torch.Tensor(data)
N = data.shape[0]
# make train/val/test
n_train = int(0.8 * N)
train_x, train_y = data[:n_train, 1:-1], data[:n_train, -1]
test_x, test_y = data[n_train:, 1:-1], data[n_train:, -1]

-------------------
-------------------

** audio_data.mat **
-------------------

data = loadmat(PATH)
train_x, test_x, train_y, test_y = data['xtrain'], data['xtest'], data['ytrain'], data['ytest']

-------------------
-------------------