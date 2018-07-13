# 画像をnumpy形式に変換するプログラム
import numpy as np
from PIL import Image
import os, glob, random
# 変数の初期化
photo_size = 75 # 画像サイズ
X = [] # 画像データを格納するリスト
y = [] # ラベルデータを格納するリスト

# path以下の画像を最大max_photoだけ読む --- (1)
def glob_images(path, label, max_photo):
  # ファイル一覧を得る --- (2)
  files = glob.glob(path + "/*.jpg")
  random.shuffle(files) # ファイルの順番をシャッフル
  # 各ファイルを処理 --- (3)
  for i, f in enumerate(files):
    if i >= max_photo: break
    # 画像ファイルを読む --- (4)
    img = Image.open(f)
    img = img.convert("RGB") # 色空間をRGBに合わせる
    # 同一サイズにリサイズ
    img = img.resize((photo_size, photo_size)) 
    # numpy形式に変換 --- (5)
    data = np.asarray(img)
    data = data / 256 # 正規化する --- (6)
    data = data.reshape(photo_size, photo_size, 3)
    X.append(data)
    y.append(label)

# 最大枚数max_photoのデータセットを作る --- (7)
def make_dataset(max_photo, outfile):
  global X
  global y
  X = []
  y = []
  # 各画像のフォルダーから写真を読み込む --- (8)
  glob_images("./25020", 0, max_photo)    # 製品 25020
  glob_images("./31500", 1, max_photo) # 製品 31500
  glob_images("./47180", 2, max_photo)      # 製品47180
  glob_images("./AF350OKNG", 3, max_photo)      # 製品AF350OK,NG
  glob_images("./B2540", 4, max_photo)      # 製品47180
  
  #glob_images("./sakura-ok", 0, max_photo)    # 桜
  #glob_images("./sunflower-ok", 1, max_photo) # ヒマワリ
  #glob_images("./rose-ok", 2, max_photo)      # バラ
  X = np.array(X, dtype=np.float32)
  np.savez(outfile, X=X, y=y)
  print("saved:" + outfile)

# データセットを2種類作成する
make_dataset(100, "photo-test4.npz")
#make_dataset(300, "photo.npz")
