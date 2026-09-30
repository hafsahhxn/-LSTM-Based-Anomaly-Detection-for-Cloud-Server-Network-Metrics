
import numpy as np

#(0,1) range.
def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))) #function 1/(1+e^-x)


class LSTM:
    def __init__(self, input_size=1, hidden_size=16, output_size=1, seed=42): 
        rng = np.random.RandomState(seed) 
         #stores dimensions as object attributes 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        z_size = input_size + hidden_size  

        def init_w(rows, cols):
            #Xavier initialization helper
            return rng.randn(rows, cols) * np.sqrt(1.0 / cols)

        # 4 LSTM Gates
        # 1) Forget gate 
        self.Wf = init_w(hidden_size, z_size)
        self.bf = np.zeros((hidden_size, 1))
        # 2) Input gate 
        self.Wi = init_w(hidden_size, z_size)
        self.bi = np.zeros((hidden_size, 1))
        # 3) Candidate cell gate 
        self.Wc = init_w(hidden_size, z_size)
        self.bc = np.zeros((hidden_size, 1))
        # 4) Output gate
        self.Wo = init_w(hidden_size, z_size)
        self.bo = np.zeros((hidden_size, 1))

        # maps final LSTM hidden state ht to predicted output 
        self.Wy = init_w(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    def forward(self, x_seq):
        
        T = x_seq.shape[0] #time steps in input sequence
        H = self.hidden_size 
        h = np.zeros((H, 1)) #hidden state
        c = np.zeros((H, 1)) #cell state

        #stores every intermediate variable at each time step
        cache = {
            "x": [], "z": [], "f": [], "i": [], 
            "cbar": [], "c": [c], "o": [], "h": [h]
        }

        #operations for LSTM t perform
        for t in range(T):
            x_t = x_seq[t].reshape(-1, 1) #input_size, 1
            z = np.vstack((h, x_t)) #hidden+input, 1

            f = sigmoid(self.Wf @ z + self.bf) #forget gate
            i = sigmoid(self.Wi @ z + self.bi) #input gate
            cbar = np.tanh(self.Wc @ z + self.bc) #candidate cell update

            c = f * c + i * cbar #update cell state
            o = sigmoid(self.Wo @ z + self.bo) #output gate
            h = o * np.tanh(c) #update hidden state

            #store step data into cache
            cache["x"].append(x_t)
            cache["z"].append(z)
            cache["f"].append(f)
            cache["i"].append(i)
            cache["cbar"].append(cbar)
            cache["c"].append(c.copy())
            cache["o"].append(o)
            cache["h"].append(h.copy())

        y_hat = self.Wy @ h + self.by  # dense layer on final hidden state
        cache["y_hat"] = y_hat
        return y_hat, cache

    def backward(self, cache, dy, learning_rate):
        H = self.hidden_size
        T = len(cache["x"])

        # Gradients accumulators
        dWf = np.zeros_like(self.Wf)
        dbf = np.zeros_like(self.bf)

        dWi = np.zeros_like(self.Wi)
        dbi = np.zeros_like(self.bi)

        dWc = np.zeros_like(self.Wc)
        dbc = np.zeros_like(self.bc)

        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)

        # Output layer gradients, uses final hidden state h_T
        h_T = cache["h"][T]
        dWy = dy @ h_T.T
        dby = dy.copy()

        dh_next = self.Wy.T @ dy   # gradient flowing into h_T from output layer
        dc_next = np.zeros((H, 1))

        for t in reversed(range(T)):
            c = cache["c"][t + 1]
            c_prev = cache["c"][t]
            
            f = cache["f"][t]
            i = cache["i"][t]
            cbar = cache["cbar"][t]
            o = cache["o"][t]
            z = cache["z"][t]

            dh = dh_next
            do_raw = (dh * np.tanh(c)) * (o * (1 - o))
            dc = dc_next + (dh * o * (1 - np.tanh(c) ** 2))
            
            dcbar_raw = (dc * i) * (1 - cbar ** 2)
            di_raw = (dc * cbar) * (i * (1 - i))
            df_raw = (dc * c_prev) * (f * (1 - f))

            dWf += df_raw @ z.T
            dbf += df_raw
            dWi += di_raw @ z.T
            dbi += di_raw
            dWc += dcbar_raw @ z.T
            dbc += dcbar_raw
            dWo += do_raw @ z.T
            dbo += do_raw

            dz = (self.Wf.T @ df_raw + 
                  self.Wi.T @ di_raw + 
                  self.Wc.T @ dcbar_raw + 
                  self.Wo.T @ do_raw)

            dh_next = dz[:H, :]
            dc_next = dc * f

        # Gradient clipping
        for grad in [dWf, dWi, dWc, dWo, dWy, dbf, dbi, dbc, dbo, dby]:
            np.clip(grad, -5.0, 5.0, out=grad)

        # SGD Update
        self.Wf -= learning_rate * dWf
        self.bf -= learning_rate * dbf
        self.Wi -= learning_rate * dWi
        self.bi -= learning_rate * dbi
        self.Wc -= learning_rate * dWc
        self.bc -= learning_rate * dbc
        self.Wo -= learning_rate * dWo
        self.bo -= learning_rate * dbo
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby

    def train_step(self, x_seq, y_true, learning_rate):
        y_hat, cache = self.forward(x_seq)
        error = y_hat - y_true.reshape(-1, 1)
        loss = float(np.mean(error ** 2))
        dy = 2 * error / self.output_size
        self.backward(cache, dy, learning_rate)
        return loss

    def predict(self, x_seq):
        y_hat, _ = self.forward(x_seq)
        return y_hat.flatten()