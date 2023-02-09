# Chess Cheater Detector, This is aimed at using a GAN to detect suspicious gameplay
import keras as k
import stockfish as fish
x_train = []
y_train = []



def generateDataset(numofmoves,exponteintallybetter,notsusmovements):
    topmoves = fish.Stockfish.get_top_moves()
    for i in range(numofmoves):
        bestmove = fish.Stockfish.get_best_move()
        x_train.append([bestmove, notsusmovements, topmoves])
        y_train.append([1, 0, 0.5, 0.5, 0.5])

def elogen(standradelo):
    for i in range(10):
        standradelo+=250
        fish.Stockfish.set_elo_rating(standradelo)
def generator(x_train,y_train,x_test,y_test):
    model = k.Sequential()
    model.add(k.Dense(64, activation='relu', input_dim=x_train.shape[1]))
    model.add(k.Dense(32, activation='relu'))
    model.add(k.Dense(16, activation='relu'))
    model.add(k.Dense(8, activation='relu'))
    model.add(k.Dense(32, activation='relu'))
    model.add(k.Dense(1, activation='sigmoid'))
    model.fit(x_train,y_train,batch_size=5,epochs=250)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=5)
    print("Model")
