!apt-get update 
!apt-get install g++ openjdk-8-jdk 
!pip install konlpy JPype1-py3 
!bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

##### 1. 데이터 로드하기 #####
urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename = "ratings_total.txt")
total_data = pd.read_table('ratings_total.txt', names = ['ratings', 'reviews'])
#print('전체 리뷰 개수: ', len(total_data)) #전체 리뷰 개수 출력

##### 2. 훈련 데이터와 테스트 데이터 분리하기 #####
total_data['label'] = np.select([total_data.ratings > 3], [1], default = 0) #레이블을 별도로 가지고 있지 않기 떄문에 평점: 4,5는 1 / 나머지는 0 부여 => 이를 label이라는 열에 저장
#print(total_data[:5])
#print(total_data['ratings'].nunique(), total_data['reviews'].nunique(), total_data['label'].nunique()) #각 열에 대해 중복을 제외한 샘플의 수 카운트
total_data.drop_duplicates(subset = ['reviews'], inplace = True) #review 열에서 중복인 내용이 있다면 중복 제거
#print('총 샘플의 수: ', len(total_data))
#print(total_data.isnull().values.any()) #Null 값이 존재하는지 확인

train_data, test_data = train_test_split(total_data, test_size = 0.25, random_state = 42) #훈련 데이터와 테스트 데이터를 3:1 비율로 분리
#print('훈련용 리뷰의 개수: ', len(train_data))
#print('테스트용 리뷰의 개수: ', len(test_data))

##### 3. 레이블의 분포 확인 #####
#train_data['label'].value_counts().plot(kind = 'bar') #막대바로 보여줌
#print(train_data.groupby('label').size().reset_index(name = 'count')) #두 레이블(1과 0) 모두 약 7만 5천개로 50:50 비율을 가지고 있음

##### 4. 데이터 정제하기 #####
train_data['reviews'] = train_data['reviews'].str.replace('[^ㄱ-하-ㅣ가-힣 ]', "") #한글과 공백을 제외하고 모두 제거
train_data['reviews'].replace('', np.nan, inplace = True)
#print(train_data.isnull().sum()) #빈 샘플이 생기지 않는지 확인

test_data.drop_duplicates(subset = ['reviews'], inplace = True) #중복 제거
test_data['reviews'] = test_data['reviews'].str.replace('[^ㄱ-하-ㅣ가-힣 ]', "") #정규 표현식 수행
test_data['reviews'].replace('', np.nan, inplace = True) #공백은 Null 값으로 변경
test_data = test_data.dropna(how = 'any') #Null 값 제거
#print('전처리 후 테스트용 샘플의 개수: ', len(test_data))

##### 5. 토큰화 #####
mecab = Mecab()
#print(mecab.morphs('와 이런 것도 상품이라고 차라리 내가 만드는 게 나을 뻔'))
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']

train_data['tokenized'] = train_data['reviews'].apply(mecab.morphs)
train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

test_data['tokenized'] = test_data['reviews'].apply(mecab.morphs)
test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

##### 6. 단어와 길이 분포 확인하기 #####
#긍정 리뷰와 부정 리뷰에 주로 어떤 단어들이 등장하는 지 빈도수를 계산함
negative_words = np.hstack(train_data[train_data.label == 0]['tokenized'].values)
positive_words = np.hstack(train_data[train_data.label == 1]['tokenized'].values)

#Counter() 함수를 통해 빈도수를 카운트
negative_word_count = Counter(negative_words)
#print(negative_word_count.most_common(20)) #부정 리뷰에 많이 등장하는 단어
positive_word_count = Counter(positive_words)
#print(positive_word_count.most_common(20)) #긍정 리뷰에 많이 등장하는 단어

#긍정/부정 리뷰에 대해 각각의 길이 분포 확인
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
text_len = train_data[train_data['label'] == 1]['tokenized'].map(lambda x: len(x))
ax1.hist(text_len, color = 'red')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('length of samples')
ax1.set_ylabel('number of samples')
#print('긍정 리뷰의 평균 길이 :', np.mean(text_len))

text_len = train_data[train_data['label'] == 0]['tokenized'].map(lambda x: len(x))
ax2.hist(text_len, color = 'blue')
ax2.set_title('Negative Reviews')
fig.suptitle('Words in texts')
ax2.set_xlabel('length of samples')
ax2.set_ylabel('number of samples')
#print('부정 리뷰의 평균 길이 :', np.mean(text_len))
plt.show()

x_train = train_data['tokenized'].values
y_train = train_data['label'].values

x_test = test_data['tokenized'].values
y_test = test_data['label'].values

##### 7. 정수 인코딩 #####
tokenizer = Tokenizer() #훈련/테스트 텍스트 데이터를 숫자로 처리할 수 있도록 정수 인코딩 수행
tokenizer.fit_on_texts(x_train)
#print(tokenizer.word_index) #각 단어에 고유한 정수가 부여됨

#등장 횟수가 1회인 단어들은 자연어 처리에서 배제
threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

#print('단어 집합(vocabulary)의 크기 :',total_cnt)
#print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
#print("단어 집합에서 희귀 단어의 비율 : ", (rare_cnt / total_cnt) * 100)
#print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율 : ", (rare_freq / total_freq) * 100)

vocab_size = total_cnt - rare_cnt + 2
#print('단어 집합의 크기: ', vocab_size)


tokenizer = Tokenizer(vocab_size, oov_token = 'OOV')
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
#print(x_train[:3])
#print(x_test[:3])

##### 8. 패딩: 서로 다른 길이의 샘플들의 길이를 동일하게 맞춰주는 작업 #####
#print('리뷰의 최대 길이 : ', max(len(review) for review in x_train))
#print('리뷰의 평균 길이 : ', sum(map(len, x_train)) / len(x_train))
plt.hist([len(review) for review in x_train], bins = 50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
      count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (count / len(nested_list)) * 100))

#리뷰의 최대 길이 = 85, 따라서 만약 80으로 패딩할 경우 몇 개의 샘플들이 보존 되는 지 확인
max_len = 80
below_threshold_len(max_len, x_train) #99퍼센트가 80 이하의 길이를 가짐 -> 따라서 훈련용 리뷰 길이 80으로 패딩

x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)

##### 9. GRU로 네이버 쇼핑 리뷰 감성 분류하기 #####
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GRU(hidden_units))
model.add(Dense(1, activation = 'sigmoid'))

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(x_train, y_train, epochs = 15, callbacks = [es, mc], batch_size = 64, validation_split = 0.2)

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(x_test, y_test)[1]))



from google.colab import drive
pet_comment = pd.read_csv('/content/긍정부정돌려볼파일.csv', encoding = 'cp949')
del pet_comment['Column1']
del pet_comment['rating']
pet_comment = pet_comment.dropna(how = 'any')
print(pet_comment)

new_pet_comment = []
new_pet_pn = []
def sentiment_predict(new_sentence):
  new_pet_comment.append(new_sentence)
  
  new_sentence = re.sub(r'[^ㄱ-하-ㅣ가-힣 ]', '', new_sentence)
  new_sentence = mecab.morphs(new_sentence)
  new_sentence = [word for word in new_sentence if not word in stopwords]
  encoded = tokenizer.texts_to_sequences([new_sentence])
  pad_new = pad_sequences(encoded, maxlen = max_len)

  score = float(loaded_model.predict(pad_new))

  if(score > 0.5):
    new_pet_pn.append('{:.2f}% 확률로 긍정 리뷰입니다.'.format(score * 100))
  else:
    new_pet_pn.append('{:.2f}% 확률로 부정 리뷰입니다.'.format((1 - score) * 100))

for i in range(0, len(pet_comment)):
  sentiment_predict(pet_comment['comment'][i])

pd_data = {'Reviews':new_pet_comment, 'P or N':new_pet_pn}
petfreinds_pd = pd.DataFrame(pd_data)
petfreinds_pd.to_csv('naver_감성분류.csv', mode = 'w')
  
print('완료')
