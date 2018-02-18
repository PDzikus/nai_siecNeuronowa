# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# funkcje aktywacji (sigmoidalna unipolarna) i jej pochodna. 
def sigmoid (x):
    return 1.0/(1.0 + np.exp(-x))

def sigDerivative (x):
    sig = sigmoid(x)
    return sig * (1-sig)

# definicja klasy SiecNeuronowa
class SiecNeuronowa(object):
    def __init__(self, warstwy):
        # generowanie losowych wartości wag (Weights) i progów (Biases) dla wszystkich warstw
        self.warstwy = warstwy
        self.base_weights = [np.random.rand(rows,cols) for cols,rows in zip(warstwy[:-1], warstwy[1:])]
        self.base_biases = [np.random.rand(size, 1) for size in warstwy[1:]]   
        self.reset()
        # funkcja aktywacji - można to zmienić na inną w razie potrzeby
        self.f = sigmoid
        self.fprime = sigDerivative
        # tablice do obliczania zmian wag i progów
        self.delta_b = [np.zeros(b.shape) for b in self.biases]
        self.delta_w = [np.zeros(w.shape) for w in self.weights]

    def reset(self):
		# reset tablicy wag i progów - przepisuje je z zapisanych wygenerowanych przy tworzeniu sieci
        self.weights = self.base_weights
        self.biases = self.base_biases
    
    def output(self, x):
	    # funkcja generująca wynik działania sieci na podstawie wejscia
        # dla każdej warstwy - funkcja aktywacji(NET = w * input + b)
        for w,b in zip(self.weights, self.biases):     
            x = self.f(np.dot(w, x) + b)
        return x
    
    def validate(self, validSet):
		# funkcja weryfikująca, sprawdza ile razu udało nam się poprawnie rozpoznać wzorce z zestawu potwierdzającego
		# zwraca liczbę rozpoznanych wzorców
        x = validSet[:,1:].T
        y = validSet[:,:1]
		# test - porównuje czy neuron o najwyższej aktywacji ma numer litery, która jest poprawną odpowiedzią
		# count - zlicza ilość poprawnych odpowiedzi
        test = np.argmax(self.output(x),axis = 0) 
        count = sum(int(i == j) for i, j in zip(test,y))
        return count
    
    # funkcja ucząca 
    def learn(self, dataSet, learningRate):
        x = dataSet[:,1:].T
        data_size = dataSet.shape[0]
		
        # macierz y będzie zawierała poprawne odpowiedzi dla całego dataSetu zakodowane jako wyjście 26 neuronów
        y = np.zeros((26,data_size),dtype = np.int16)
        answers = dataSet[:,0]
        for idy in range(data_size):
            y[int(answers[idy]),idy] = 1

        # przepuszczamy dane wejsciowe przez kolejne warstwy sieci zapamietujac wyniki operacji
		# nets - suma ważona W*X + b (przed funkcją aktywacji)
        nets = []
        activations = [x]
        for b, w in zip(self.biases, self.weights):
            x = np.dot(w, x) + b
            nets.append(x)
            x = self.f(x)
            activations.append(x)
 
        # wyliczamy błędy dla wszystkich odpowiedzi i robimy wsteczną propagację błedu
        # pierwsza (od końca) warstwa
        error = (y - activations[-1]) * self.fprime(nets[-1])
        self.delta_b[-1] = np.dot(error,np.ones((data_size, 1))) * learningRate 
        self.delta_w[-1] = np.dot(error, activations[-2].T) * learningRate
        # druga warstwa - wsteczna propagacja błędu:
        error = np.dot(self.weights[-1].T, error) * self.fprime(nets[-2])
        self.delta_b[-2] = np.dot(error, np.ones((data_size,1))) * learningRate
        self.delta_w[-2] = np.dot(error, activations[-3].T) * learningRate

        # ostatnia faza uczenia - wyliczamy nowe wagi
        self.weights = [w + dw for w, dw in zip(self.weights, self.delta_w)]
        self.biases = [b + db for b, db in zip(self.biases, self.delta_b)]

########################################################################################################
# przygotowanie danych
########################################################################################################
df = pd.read_csv('letter-recognition.data', header = None)

# konwersja liter do kolejnych numerów - A = 0, Z = 25
df[0] = df[0].apply(lambda x: int(ord(x) - ord('A')))

# dzielimy zbiór danych na dwa podzbiory: train set = 75% wszystkich danych, validation set = 25%
# wszystkie dane wejściowe są standaryzowane
data_size = df.shape[0]
train_size = int(data_size * 75/100)
valid_size = data_size - train_size
standard_df = (df-df.mean())/df.std()
standard_df[0] = df[0]
trainSet = standard_df.loc[:train_size-1,:].values
validSet = standard_df.loc[train_size:,:].values

# domyślne parametry sieci 
neuronsInput = 16
neuronsHidden = 65   # ten parametr będziemy modyfikować
neuronsOutput = 26

# parametry uczenia się:
learningRate = 0.1    
epochs = 500
batch_size = 250

# Główna pętla ucząca - dla każdej epoki generuje kolejną permutację danych uczących
# grupuje je w mini_zestawy i wywołuję funkcję uczącą dla kolejnych minizestawów
# Dla każdego eksperymentu generujemy wydruk: % poprawnie rozpoznanych wzorców w kolejnych epokach (co 20)
##########################################################################################################
x = np.arange(0,epochs + 1,20)
for neuronsHidden in [ 52, 78, 104 ]:
	# dla każdej wartości neuronsHidden generujemy nową sieć neuronową
	layer_sizes = [neuronsInput,neuronsHidden, neuronsOutput]
	NN = SiecNeuronowa(layer_sizes)
	for batch_size in [ 50, 100, 200 ]:
		# wykres (eksperyment) będzie generował porównanie precyzji dla różnych wartości LR
		# przy ustalonym batch_size
		eksperyment = []
		for learningRate in [ 0.1, 0.01, 0.005 ]:	
			# dla każdej kombinacji learning Rate i batch_size resetujemy wagi sieci do zapamiętanych
			# żeby zmniejszyć wpływ losowych wag na wyniki
			NN.reset()
			print('Parametry sieci: Neurony: {0}, LR {1}, Batch: {2}'.format(neuronsHidden, learningRate, batch_size))
			pomiar = []
			np.random.seed(1)
			for epoka in range(epochs):
				# co 20 epok sprawdzamy precyzję (w procentach) sieci
				if epoka % 20 == 1:
					pomiar.append(NN.validate(validSet)/(valid_size/100))
					print('Epoka {0}/{1}: {2}%'.format(epoka,epochs,pomiar[-1]))
				# dane treningowe są mieszane i dzielone na batche o ustalonej wielkości
				np.random.shuffle(trainSet)
				mini_batches = [trainSet[k:k+batch_size] for k in range(0, 15000, batch_size)]
				for mini_batch in mini_batches: 
					NN.learn(mini_batch, learningRate)			
			# dodajemy finalny pomiar po ostatniej epoce uczenia i zapisujemy wynik eksperymentu
			pomiar.append(NN.validate(validSet)/50)
			eksperyment.append([learningRate,pomiar])
			
		# po wykonaniu testów dla wszystkich wartości learningRate robimy wykres wyników
		# i zapisujemy go na dysku (plt.savefig) albo wyświetlamy (plt.show)
		for zestaw in eksperyment:
			plt.plot(x, zestaw[1], label = 'LR {0} : {1}%'.format(zestaw[0], zestaw[1][-1]))
		plt.xlabel('Epoki uczenia')
		plt.ylabel('% poprawnie rozpoznanych liter')
		plt.title('Parametry sieci: Neurony: {0}, Batch: {1}'.format(neuronsHidden, batch_size))
		plt.axis([0,epochs, 0, 100])
		plt.grid()
		plt.legend()
		plt.savefig('neurons{0}_B{1}.png'.format(neuronsHidden, batch_size), bbox_inches='tight', dpi = 1000)
		#plt.clf()
		plt.show() 


