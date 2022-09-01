import os
import haiku as hk
import matplotlib.pyplot as plt

import jax
from jax import value_and_grad
import tqdm
import jax.numpy as jnp
import numpy as np

class Actor(hk.Module):
    def __init__(self, s_dim):
        super().__init__(name="CNN")
        self.s_dim = s_dim
        self.linear1 = hk.Linear(100)
        self.linear2 = hk.Linea(10)
        self.linear3 = hk.Linear(1)

    def __call__(self, x_batch):
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        x = jax.nn.relu(x)
        x = self.linear3(x)
        return x

class Critic(hk.Module):
    def __init__(self, a_dim, s_dim):
        super().__init__(name="CNN")
        self.s_dim = s_dim
        self.a_dim = a_dim
        
        self.linear1 = hk.Linear(100)
        self.linear2 = hk.Linea(10)
        self.linear3 = hk.Linear(1)

    def __call__(self, x_batch):
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        x = jax.nn.relu(x)
        x = self.linear3(x)
        return x



def actorNet(x):
    cnn = Actor()
    return cnn(x)

def CrossEntropyLoss(weights, input_data, actual):
    preds = conv_net.apply(weights, rng, input_data)
    one_hot_actual = jax.nn.one_hot(actual, num_classes=len(classes))
    log_preds = jnp.log(preds)
    return - jnp.sum(one_hot_actual * log_preds)

conv_net = hk.transform(actorNet)
def UpdateWeights(weights,gradients):
    return weights - learning_rate * gradients


rng = jax.random.PRNGKey(42) ## Reproducibility ## Initializes model with same weights each time.

conv_net = hk.transform(actorNet)
params = conv_net.init(rng, X_train[:5])
epochs = 25
batch_size = 256
learning_rate = jnp.array(1/1e4)

with tqdm.tqdm(range(1, epochs+1)) as pbar:

    for i in pbar:
        batches = jnp.arange((X_train.shape[0]//batch_size)+1) ### Batch Indices

        losses = [] ## Record loss of each batch
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
            else:
                start, end = int(batch*batch_size), None

            X_batch, Y_batch = X_train[start:end], Y_train[start:end] ## Single batch of data

            loss, param_grads = value_and_grad(CrossEntropyLoss)(params, X_batch, Y_batch)
            #print(param_grads)
            params = jax.tree_map(UpdateWeights, params, param_grads) ## Update Params
            losses.append(loss) ## Record Loss
            
            if i%10 == 0:
                save_model(params)

        pbar.set_description("CrossEntropy Loss : {:.3f}".format(jnp.array(losses).mean()))

def MakePredictions(weights, input_data, batch_size=32):
    batches = jnp.arange((input_data.shape[0]//batch_size)+1) ### Batch Indices

    preds = []
    for batch in batches:
        if batch != batches[-1]:
            start, end = int(batch*batch_size), int(batch*batch_size+batch_size)
        else:
            start, end = int(batch*batch_size), None

        X_batch = input_data[start:end]

        preds.append(conv_net.apply(weights, rng, X_batch))

    return preds

train_preds = MakePredictions(params, X_train, 256)
train_preds = jnp.concatenate(train_preds).squeeze()
train_preds = train_preds.argmax(axis=1)

test_preds = MakePredictions(params, X_test, 256)
test_preds = jnp.concatenate(test_preds).squeeze()
test_preds = test_preds.argmax(axis=1)


print("Train Accuracy : {:.3f}".format(accuracy_score(Y_train, train_preds)))
print("Test  Accuracy : {:.3f}".format(accuracy_score(Y_test, test_preds)))

print("Test Classification Report ")
print(classification_report(Y_test, test_preds))

plt.plot(losses)
plt.savefig(os.path.join('plots', 'loss.jpg'))

