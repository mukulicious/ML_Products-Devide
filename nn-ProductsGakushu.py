from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.utils import np_utils
import numpy as np

#グラフ用に追加
import matplotlib.pyplot as plt

  #学習結果をグラフで表示するための関数
def plot_history(history):
    print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


# 変数の宣言 --- (1)
classes = 5 # いくつに分類するか
data_size = 75 * 75 * 3 # 縦75×横75×3原色

# データを学習しモデルを評価する
def main():
  # データの読み込み --- (2)
  data = np.load("./photo-test4.npz")
  X = data["X"] # --- 画像データ
  y = data["y"] # --- ラベルデータ
  # データを2次元に変形する --- (3)
  X = np.reshape(X, (-1, data_size))
  # 訓練データとテストデータに分割 --- (4)
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  # モデルを訓練し評価 --- (5)
  model,history = train(X_train, y_train,X_test,y_test)
  model_eval(model, X_test, y_test)
  
  #学習データのグラフ表示
  plot_history(history)

# モデルを構築しデータを学習する
def train(X, y,testData_x,testData_y):
  # モデルの構築 --- (6)
  model = Sequential()
  model.add(Dense(units=64, input_dim=(data_size))) # --- (6.1)
  model.add(Activation('relu')) # --- (6.2)
  model.add(Dense(units=classes)) # --- (6.3)
  model.add(Activation('softmax')) # --- (6.4)
  model.compile(loss='sparse_categorical_crossentropy',
        optimizer='sgd', metrics=['accuracy'])
  #model.fit(X, y, epochs=60) # データを学習 --- (7)
  # データを学習 --- (7) #グラフ表示の時は返値をもらう
  #history = model.fit(X, y,
  #                  batch_size=128,
  #                  epochs=60,
  #                  verbose=1,
  #                  validation_data=(testData_x, testData_y))
  
  #historyはグラフ表示の為にに使用。　グラフ表示しない場合は不要
  history = model.fit(X, y,
                    epochs=60,
                    verbose=1,
                    validation_data=(testData_x, testData_y))
    #引数 verbose=0は計算結果詳細の表示設定 0:表示しない 1:表示する　2:進捗バーなし
  
  #学習データの保存
  model.save_weights("Products_test4.hdf5")
  
  return model,history

# モデルを評価する 
def model_eval(model, X_test, y_test):
  score = model.evaluate(X_test, y_test,verbose=0) # --- (8)
  #引数 verbose=0は計算結果詳細の表示設定 0:表示しない 1:表示する
  print('loss=', score[0])
  print('accuracy=', score[1])
  
if __name__ == "__main__":
  main()
  


