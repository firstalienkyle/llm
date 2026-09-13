import numpy as np
import ttokenizer as pt

'''
hf_token = input("Please enter the hf token:")
config = DownloadConfig(token=hf_token)

segment = load_dataset('HuggingFaceFW/fineweb-edu', name='default', split='train', streaming=True, download_config=config)

current_segment = iter(segment)
text = next(current_segment)
'''

text ="The 2008 Scottish Labour Party leadership election was an internal party election to choose a new leader of the Labour Party in the Scottish Parliament, and was triggered following the resignation of Wendy Alexander following a row over donations to her own leadership campaign in 2007.[1] Iain Gray won the contest and was announced as leader on 13 September 2008.It was the second Scottish Labour leadership election in as many years, the first being caused by the resignation of Jack McConnell,[2] following the Scottish National Party's victory over Labour in the 2007 Scottish Parliament election, however in this election, Alexander was unopposed, meaning that no ballot actually took place.[3]The timetable for the election was finalised on Monday 28 July, having been put on hold for a month to allow the party to focus on the Glasgow East by-election,[4] which ultimately saw the Scottish National Party overturn a 13,507 Labour majority to gain the seat.[5] Nominations closed at noon on Friday 1 August with the result being declared on Saturday 13 September.A deputy leadership election was held alongside the leadership election following the resignation of Cathy Jamieson on 28 July. Johann Lamont was elected deputy leader."

epochs = 1000

number_points = 500
number_class = 3

layers = 4
layer_size = 64
inputs = 2
outputs = number_class

try:
    weights_archive = np.load("/Users/meimozhu/Desktop/code/python/ai/mine/array_weights.npz")
    biases_archive = np.load("/Users/meimozhu/Desktop/code/python/ai/mine/array_biases.npz")
    HAS_SAVED_DATA = True
except FileNotFoundError:
    HAS_SAVED_DATA = False

# Temperary 
HAS_SAVED_DATA = False

'''
def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y
'''

section = 1

def next_text(section):
    enc_text = pt.merge(pt.encode(text))
    return pt.embed((enc_text[:section-1],enc_text[section]))

X,y = next_text(section)
    
class Layer_D:
    def __init__(self, n_ins, n_neurons, name):
        self.name = name
        if HAS_SAVED_DATA:
            self.weights = weights_archive[f'{self.name}_weights']
            self.biases = biases_archive[f'{self.name}_biases']
        else:
            self.weights = 0.1 * np.random.randn(n_ins, n_neurons)
            self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def ReLU(self):
        self.output = np.maximum(0, self.output)

    def softmax(self):
        exp_array = np.exp(self.output - np.max(self.output, axis=1, keepdims=True))
        self.output = exp_array / np.sum(exp_array, axis=1, keepdims=True)

class Muon:
    def __init__(self, lr=0.05, mom=0.9):
        self.lr = lr
        self.mom = mom
        self.vel_w = {}
        self.vel_b = {}

    def update_layer(self, layer, layer_id):
        if layer_id not in self.vel_w:
            self.vel_w[layer_id] = np.zeros_like(layer.weights)
            self.vel_b[layer_id] = np.zeros_like(layer.biases)

        self.vel_w[layer_id] = self.mom * self.vel_w[layer_id] - self.lr * layer.dweights
        self.vel_b[layer_id] = self.mom * self.vel_b[layer_id] - self.lr * layer.dbiases

        layer.weights += self.vel_w[layer_id]
        layer.biases += self.vel_b[layer_id]

def forward_pass(X, layer_1, layer_2, layer_3, layer_4):
    layer_1.forward(X)
    layer_1.ReLU()
    layer_2.forward(layer_1.output)
    layer_2.ReLU()
    layer_3.forward(layer_2.output)
    layer_3.ReLU()
    layer_4.forward(layer_3.output)
    layer_4.softmax()

def backward_pass(X, y, layer_1, layer_2, layer_3, layer_4):
    samples = len(y)

    dvalues = layer_4.output.copy()
    dvalues[range(samples), y] -= 1
    dvalues /= samples

    layer_4.dweights = np.dot(layer_3.output.T, dvalues)
    layer_4.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    dinputs = np.dot(dvalues, layer_4.weights.T)

    dinputs[layer_3.output <= 0] = 0
    layer_3.dweights = np.dot(layer_2.output.T, dinputs)
    layer_3.dbiases = np.sum(dinputs, axis=0, keepdims=True)
    dinputs = np.dot(dinputs, layer_3.weights.T)

    dinputs[layer_2.output <= 0] = 0
    layer_2.dweights = np.dot(layer_1.output.T, dinputs)
    layer_2.dbiases = np.sum(dinputs, axis=0, keepdims=True)
    dinputs = np.dot(dinputs, layer_2.weights.T)

    dinputs[layer_1.output <= 0] = 0
    layer_1.dweights = np.dot(X.T, dinputs)
    layer_1.dbiases = np.sum(dinputs, axis=0, keepdims=True)

'''
X, y = spiral_data(number_points, number_class)
'''

layer_1 = Layer_D(2, 64, "layer_1")
layer_2 = Layer_D(64, 64, "layer_2")
layer_3 = Layer_D(64, 64, "layer_3")
layer_4 = Layer_D(64, number_class, "layer_4")

if HAS_SAVED_DATA:
    weights_archive.close()
    biases_archive.close()

array_weights = [getattr(globals()[f'layer_{i+1}'], 'weights') for i in range(layers)]
array_biases = [getattr(globals()[f'layer_{i+1}'], 'biases') for i in range(layers)]

optimizer = Muon(lr=0.05, mom=0.9)

best_loss = 9999999999999
best_weights = {}
best_biases = {}

for epoch in range(epochs + 1):
    forward_pass(X, layer_1, layer_2, layer_3, layer_4)

    section += 1
    next_text(section)

    correct_confidences = layer_4.output[range(len(layer_4.output)), y]
    loss = np.mean(-np.log(np.clip(correct_confidences, 1e-7, 1 - 1e-7)))
    predictions = np.argmax(layer_4.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < best_loss:
        best_loss = loss
        best_weights = {
            'layer_1': layer_1.weights.copy(),
            'layer_2': layer_2.weights.copy(),
            'layer_3': layer_3.weights.copy(),
            'layer_4': layer_4.weights.copy()
        }
        best_biases = {
            'layer_1': layer_1.biases.copy(),
            'layer_2': layer_2.biases.copy(),
            'layer_3': layer_3.biases.copy(),
            'layer_4': layer_4.biases.copy()
        }

    backward_pass(X, y, layer_1, layer_2, layer_3, layer_4)
    optimizer.update_layer(layer_1, 1)
    optimizer.update_layer(layer_2, 2)
    optimizer.update_layer(layer_3, 3)
    optimizer.update_layer(layer_4, 4)

    if epoch % max(1, epochs // 10) == 0:
        print(f"epoch: {epoch}, loss: {loss:.4f}, accuracy: {accuracy:.4f}")

layer_1.weights, layer_1.biases = best_weights['layer_1'], best_biases['layer_1']
layer_2.weights, layer_2.biases = best_weights['layer_2'], best_biases['layer_2']
layer_3.weights, layer_3.biases = best_weights['layer_3'], best_biases['layer_3']
layer_4.weights, layer_4.biases = best_weights['layer_4'], best_biases['layer_4']

forward_pass(X, layer_1, layer_2, layer_3, layer_4)
predictions = np.argmax(layer_4.output, axis=1)
final_loss = np.mean(-np.log(np.clip(layer_4.output[range(len(layer_4.output)), y], 1e-7, 1 - 1e-7)))
final_accuracy = np.mean(predictions == y)

print(f"final loss (best run): {final_loss:.4f}")
print(f"final accuracy (best run): {final_accuracy:.4f}")

output = pt.decode(layer_4.output)

'''
array_weights = [layer_1.weights, layer_2.weights, layer_3.weights, layer_4.weights]
array_biases = [layer_1.biases, layer_2.biases, layer_3.biases, layer_4.biases]

array_weights_dict = {f'layer_{i}_weights': arr for i, arr in enumerate(array_weights, start=1)}
np.savez_compressed("/Users/meimozhu/Desktop/code/python/ai/mine/array_weights.npz", **array_weights_dict)

array_biases_dict = {f'layer_{i}_biases': arr for i, arr in enumerate(array_biases, start=1)}
np.savez_compressed("/Users/meimozhu/Desktop/code/python/ai/mine/array_biases.npz", **array_biases_dict)
'''

'''
del segment
del current_segment
'''
