# LSTM Model
> 데이터셋 :  **Netflix Review** [¶](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated)<br>
[¶ 데이터 전처리](#데이터-전처리)<br>
[¶ 시퀀스 데이터 변환](#시퀀스-데이터-변환)<br>
[¶ 데이터 세팅](#데이터-세팅)<br>
[¶ 모델 정의](#lstm-모델-정의)<br>
[¶ 예측 함수](#예측-함수---pipeline)<br>
[¶ 파라미터 조정](#파라미터-조정)<br>


# 데이터 전처리
## 불러오기

```py
df[df['userName'].isnull()]   # userName 열의 결측치 확인하기
df['userName'] = df['userName'].fillna('anonymous')   # userName 열의 결측치를 anonymous로 채우기

print(df['score'].min())   # score 최솟값 확인하기
print(df['score'].max())   # score 최댓값 확인하기

df[df['content'].isnull()]   # content 열의 결측치 확인하기
# content 열에 결측치가 있는 경우 score에 따라 내용을 채우는 함수 생성하기
def content_filled(row):
    if pd.isnull(row['content']):
        if row['score'] == 5:
            return 'I love it.'
        elif row['score'] == 4:
            return 'I like it.'
        elif row['score'] == 3:
            return 'So-so.'
        elif row['score'] == 2:
            return 'Not good.'
        elif row['score'] == 1:
            return 'Bad.'
    else:
        return row['content']

# content 열의 결측치 처리하기
df['content'] = df.apply(content_filled, axis=1)

df[df['reviewCreatedVersion'].isnull()]   # 결측치 확인하기
df['reviewCreatedVersion'] = df['reviewCreatedVersion'].ffill()   # ffill() : 결측값을 바로 위 값과 동일하게 변경

df[df['appVersion'].isnull()]   # appVersion 열의 결측치 확인하기
df['appVersion'] = df['appVersion'].ffill()   # ffill로 결측치 채우기

df.isnull().sum()
```

## 결측치 처리

```py
df.isnull().sum()
#
reviewId                    0
userName                    2
content                     2
score                       0
thumbsUpCount               0
reviewCreatedVersion    17488
at                          0
appVersion              17488
dtype: int64

df['content'] = df['content'].fillna("so so")
```

## 중복 처리

```py
df.duplicated(subset=['reviewId', 'userName', 'content', 'score']).sum() # 337
df = df.drop_duplicates(subset=['reviewId', 'userName', 'content', 'score'], keep = 'first')
# 첫번째만 남김
df.duplicated(subset=['reviewId', 'userName', 'content', 'score']).sum() # 0
```

## 불용어 처리
1. 숫자, 문장부호, 이모지 제거 : 문자만 추출
2. 불필요한 품사 제거 : 고유명사, 관사 등
3. 단어 형태 통일
4. 대체가 필요한 단어 대체
5. 철자 검사

```py 
# 불용어 설정
stop_words = set(stopwords.words('english'))
additional_stopwords = {'app', 'netflix', 'show', 'time', 'series', 'phone', 'movie', 'tv', 'would', 'watch'}
stop_words = stop_words.union(additional_stopwords)
stop_words.discard('not')
# not 이 빠지면 의미가 달리지는 경우가 있어 불용어에서 제외

# 표제어 추출 
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def lemma_v(sentence): # 동사
    return [lemmatizer.lemmatize(word, 'v') for word in sentence] 

def lemma_n(sentence): # 명사
    return [lemmatizer.lemmatize(word, 'n') for word in sentence]


#불필요한 품사[¶](#pos-태그-목록) 제거 

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def TagFitter(sentense):
  tokens = pos_tag(word_tokenize(sentense)) # 토큰화
  tags = [t[0] for t in tokens if t[1] != "NNP" and t[1] != "NNPS" and t[1] != "PRP" and t[1] != "DT"]
  # 특정 품사에 해당하지 않는 단어추출
  return ' '.join(tags) # 문장으로 복원


# 단어 대체

def t_replacer(sentense):
  tokens = word_tokenize(sentense)
  r = []
  for token in tokens:
    if token == 't': 
      r.append('not')
    else: 
      r.append(token)
  return ' '.join(r)

# 철자 검토 함수 

from textblob import TextBlob
def WordCorrect(senntense):
  s = TextBlob(senntense)
  s = s.correct()
  s = ''.join(s)
  return s

# 전처리 함수

# 과도한 전처리를 할 경우 데이터 수가 줄어들어 성능 저하 - 여러번 테스트하며 조절
def preprocessing(sentence):
  if isinstance(sentence, float): return ''
  cleaned = re.sub('[^a-zA-Z]', ' ', sentence)
  # cleaned = t_replacer(cleaned)
  # cleaned = TagFitter(cleaned)
  cleaned = cleaned.lower()
  cleaned = cleaned.strip()
  cleaned = cleaned.split()
  cleaned = lemma_v(cleaned)
  cleaned = lemma_n(cleaned)
  # cleaned = [word for word in cleaned if word not in stop_words]
  cleaned = ' '.join(cleaned)
  # cleaned = WordCorrect(cleaned)
  return cleaned

content[0:5].apply(preprocessing)
  # 
0                                           can t open
1                                         the best app
2    most of the famous korean drama be not dub in ...
3    it s superb but can you please add comment sec...
4    the only reason i didn t give it four star be ...
Name: content, dtype: object
```
## 전처리 함수 적용
- lambda 활용 : 명확한 범위 지정 필요

```py
df.loc[:, 'content'] = df['content'].apply(lambda x: preprocessing(x))
```

## 데이터 준비

```py
reviews = df['content']
ratings = df['score']
if len(reviews) != len(ratings):
    print("warning: do not match")
```

## Feature 분석

```py
import seaborn as sns
import matplotlib.pyplot as plt
# 그래프 스타일 설정
sns.set(style='whitegrid')

# 점수의 빈도 수를 계산하여 데이터프레임으로 변환
score_counts = df['score'].value_counts().reset_index()
score_counts.columns = ['score', 'count']

plt.figure(figsize=(8, 6))  # 그래프 크기 설정
sns.barplot(x='score', y='count', data=score_counts, hue='score', palette='viridis', legend=False)    # 막대 그래프 그리기

# 점수의 빈도 수 계산:
# value_counts(): 이 메서드는 지정된 열의 고유한 값과 그 빈도를 계산하여 반환하는 시리즈 객체이다.
# df['score'].value_counts(): 각 점수의 빈도를 계산한다.
# .reset_index(): 인덱스를 초기화하여 새로운 데이터프레임으로 변환한다.
# score_counts.columns = ['score', 'count']: reset_index로 초기화한 열 이름을 다시 지정한다.

# 그래프 제목 및 축 레이블 추가
plt.title('Distribution of Scores', fontsize=16)
plt.xlabel('Score', fontsize=14)
plt.ylabel('Count', fontsize=14)

# 그래프 출력
plt.show()
```

![](/team_10_LSTM/images/bar_1.png)

### 그래프 종류

1. 선 그래프 (Line Plot): 데이터를 선으로 연결한 그래프.
2. 막대 그래프 (Bar Plot): 범주형 데이터를 막대로 표현한 그래프.
3. 산점도 (Scatter Plot): 두 변수 간의 관계를 점으로 표현한 그래프.
4. 히스토그램 (Histogram): 데이터 분포를 막대의 높이로 표현한 그래프.
5. 파이 차트 (Pie Chart): 각 항목의 비율을 원형으로 표현한 그래프.
6. 상자 그림 (Box Plot): 데이터의 사분위수를 보여주는 그래프.
7. 히트맵 (Heatmap): 값의 크기를 색상으로 표현한 그래프.
8. 커널 밀도 추정 (KDE Plot): 데이터의 분포를 곡선으로 표현한 그래프.

### 선 그래프 (Line Plot)

- 데이터의 빈도를 계산하여 선 그래프로 그리기
1. 데이터의 빈도 계산하기
- `score_counting= df['score'].value_counts().sort_index()`
- `plt.plot(score_counting.index, score_counting.values)`
- `plt.title("Line Plot of Score Distribution")`# 그래프의 제목을 설정한다. 이 경우 "Score Distribution"이라는 제목이 그래프 상단에 표시된다.
- p`lt.xlabel("Score")# plt.xlabel()`: x축의 레이블을 설정한다. 이 경우 x축에 "Score"라는 레이블이 붙는다.
- `plt.ylabel("Count")# plt.ylabel()`: y축의 레이블을 설정한다. 이 경우 y축에 "Frequency"라는 레이블이 붙다.
- `plt.show()`
1. plt와 sns 차이점
`plt.plot()` (matplotlib): 기본적인 선 그래프를 그리며, 레이블, 축 설정 등을 수동으로 추가해야 한다.
`sns.lineplot()` (seaborn): plt.plot()보다 더 높은 수준의 API로, 자동으로 축 레이블과 스타일링이 추가되고, DataFrame과의 통합이 더 쉽다.

![](/team_10_LSTM/images/line_1.png)

### 막대 그래프 (Bar Plot)

- 데이터의 빈도를 계산하여 막대 그래프로 그리기
1. 그래프 그리기
- `plt.bar(score_counting.index, score_counting.values)`
- `plt.title("Bar Plot of Score Distribution")`
- `plt.xlabel("Score")`
- `plt.ylabel("Count")`
- `plt.show()`
1. plt와 sns 차이점
- `plt.bar() (matplotlib)`: 단순한 막대 그래프 그리기, 축 레이블과 스타일을 수동으로 설정해야 한다.
- `sns.barplot() (seaborn)`: mean과 confidence interval이 자동으로 계산되어 표시되며 DataFrame을 쉽게 사용해 그룹별 평균을 시각화할 수 있다.

![](/team_10_LSTM/images/bar_2.png)

### 산점도 (Scatter Plot)

1. 데이터의 빈도를 계산하여 산점도로 그리기
- `plt.scatter(score_counting.index, score_counting.values)`
- `plt.title("Scatter Plot of Score Distribution")`
- `plt.xlabel("Score")`
- `plt.ylabel("Count")`
- `plt.show()`
1. plt와 sns 차이점
- `plt.scatter() (matplotlib)`: 산점도를 그릴 때 수동으로 색, 크기 등을 지정해야 한다.
- `sns.scatterplot() (seaborn)`: hue, size, style 인수를 통해 카테고리별로 색깔과 모양을 자동으로 구분해준다.# 데이터프레임과 함께 쉽게 사용 가능하다.

![](/team_10_LSTM/images/scatter.png)

### 히스토그램 (Histogram)

1. 히스토그램으로 데이터 분포 그리기
- `plt.hist(df['score'], bins=5, range=(1, 5), edgecolor='black')`
- `plt.title("Histogram of Score Distribution")`
- `plt.xlabel("Score")`
- `plt.ylabel("Count")`
- `plt.show()`
1. plt와 sns 차이점
- `plt.plot() (matplotlib)`: 기본적인 선 그래프를 그리며, 레이블, 축 설정 등을 수동으로 추가해야 한다.
- `sns.lineplot() (seaborn)`: plt.plot()보다 더 높은 수준의 API로, 자동으로 축 레이블과 스타일링이 추가되고, DataFrame과의 통합이 더 쉽다.

![](/team_10_LSTM/images/histo.png)

### 파이 차트 (Pie Chart)

1. 데이터의 빈도를 계산하여 파이 차트로 그리기
- `score_counts_not_sortedindex= df['score'].value_counts()`
- `plt.pie(score_counts_not_sortedindex, labels=score_counts.index, autopct='%1.1f%%')`
- `plt.title("Pie Chart of Score Distribution")`
`plt.show()`
1.  plt와 sns 차이점
- `plt.pie()`: matplotlib에서만 제공하고, 직접 수동 설정이 필요하다.
- sns: Seaborn은 파이 차트를 제공하지 않는다.

![](/team_10_LSTM/images/pie.png)

### 상자 그림 (Box Plot) :

1. 상자 그림으로 데이터 분포 그리기
- `plt.boxplot(df['score'])`
- `plt.title("Box Plot of Score Distribution")`
- `plt.ylabel("Score")`
- `plt.show()`
1.  plt와 sns 차이점
- `plt.boxplot()`: 한 번에 여러 개의 데이터 세트를 위치 인수로 받을 수 있다.
- `sns.boxplot()`: 하나의 데이터 세트만 위치 인수로 받을 수 있으며, x 또는 y를 지정해준다.
- 그러나 데이터프레임과 함께 사용하여 x와 y를 축별로 나누어 사용할 수 있다.

![](/team_10_LSTM/images/box.png)

### 히트맵 (Heatmap)

- 범주형 데이터이므로 빈도에 대한 히트맵
1.  점수별 빈도를 데이터프레임으로 변환
- `score_counts_frame= df['score'].value_counts().sort_index().to_frame().T`
- **T는 데이터프레임을 전치(transpose)하는 메서드**
- `sns.heatmap(score_counts_frame, annot=True, cmap="YlGnBu", cbar=False)`
- `plt.title("Heatmap of Score Distribution")`
- `plt.xlabel("Score")`
- `plt.show()`
1. plt와 sns 차이점
- `plt.imshow()`: 이미지 또는 행렬 데이터를 단순히 시각화하는 데 적합하고, 데이터프레임 통합이 어렵고, 수동 설정이 많다.
- `sns.heatmap()`: 데이터프레임과의 통합이 매우 쉽고, 상관 행렬 등의 시각화에 유용하며, 자동화된 스타일링을 제공한다.

![](/team_10_LSTM/images/heat.png)

### 커널 밀도 추정 (KDE Plot)

1. KDE 플롯으로 데이터 분포 그리기
- `sns.kdeplot(df['score'], bw_adjust=0.5)`
- `plt.title("KDE Plot of Score Distribution")`
- `plt.xlabel("Score")`
- `plt.ylabel("Density")`
- `plt.show()`
1. plt와 sns 차이점
- `plt.plot()과 scipy.stats.kde()` (matplotlib): 커널 밀도 추정을 수동으로 계산하고 그려야 한다
- `sns.kdeplot()` (seaborn): kde 그래프를 간단히 그릴 수 있으며,
- 여러 카테고리나 변수를 쉽게 시각화할 수 있다.

![](/team_10_LSTM/images/kde.png)

# 시퀀스 데이터 변환
## 토큰화 
- 시퀀스 데이터로 변형하기 전에 문장을 끊어서 토큰화
- 토큰화를 하면 가공하기 쉬워짐

### TorchText 예시

```py
from torchtext.data.utils import get_tokenizer # 토큰화 모듈
from torchtext.vocab import build_vocab_from_iterator # 단어집 생성 모듈

# 텍스트 데이터 토큰화 = 단어 단위로 분할
tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
  for text in data_iter:
    yield tokenizer(text)
    # data_iter: 반복 가능한 객체
    # yield: 제너레이터, 호출될 때 마다 반환(메모리 절약, 큰 데이터셋 처리시)

vocab = build_vocab_from_iterator(yield_tokens(reviews), specials=['<unk>'])
# vocab 클래스 생성
# 단어 집합 생성 = 가공되지 않은 학습 데이터셋으로 어휘집 생성 
# specials=["<unk>"] = 어휘에 포함되지 않은 단어도 특정 토큰으로 처리

vocab.set_default_index(vocab['<unk>'])
# 어휘에 없는 단어가 있을 때 사용할 인덱스 설정
```

### tensorflow 예시

```py
from tensorflow.keras.preprocessing.text import Tokenizer
 
tokenizer = Tokenizer(oov_token='<OOV>') # 없는 데이터 치환
tokenizer.fit_on_texts(reviews) # 단어집 생성
len(tokenizer.word_index) # 단어집 개수
```

## Padding
- 정수형 데이터로 변환 
- 빈 부분 0으로 페딩 처리
### TorchText 예시

```py
from torch.nn.utils.rnn import pad_sequence

max_length = 50
numericalized_data = []

for text in reviews:
    indices = [vocab[token] for token in tokenizer(text)]
    if len(indices) > max_length:
        # 길이가 초과할 경우 뒷부분 잘라내기
        indices = indices[:max_length]
    elif len(indices) < max_length:
        # 길이가 모자랄 경우 0으로 패딩하기
        indices += [0] * (max_length - len(indices))
    numericalized_data.append(torch.tensor(indices))

# 시퀀스 패딩 (이미 길이가 맞춰졌으므로 패딩 필요 없음)
padded_data = pad_sequence(numericalized_data, batch_first=True)

# NumPy 배열로 변환
reviews = padded_data.numpy()

print(reviews)
# 출력 결과는 다를 수 있음, 형태만 참고
array([ 155,   43,  816,   58,  767,   52,  565,  130,  287,   12,  286,
        167,   27,  311,  100,   15,   32,  575, 1393,  307,  901, 1046,
         26,  419,  189,   33,  501,  780,  207,   83,   69, 1071,   21,
         21,   72,  567,  287,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0], dtype=int32)
```

### Tensorflow 예시

```py
from tensorflow.keras.preprocessing.sequence import pad_sequences
reviews = tokenizer.texts_to_sequences(reviews)

MAX_LENGTH = 50 # 최대 문장의 길이
TRUNC = 'post' # 넘칠 경우 자르기 / 앞부분 = pre, 뒷부분 = post
PAD = 'post' # 모자랄 경우 채우기(0) / 위와 같음
reviews = pad_sequences(reviews_seq, maxlen= MAX_LENGTH, truncating = TRUNC, padding = PAD)
reviews[4]
# 출력 결과는 다를 수 있음, 형태만 참고
array([ 155,   43,  816,   58,  767,   52,  565,  130,  287,   12,  286,
        167,   27,  311,  100,   15,   32,  575, 1393,  307,  901, 1046,
         26,  419,  189,   33,  501,  780,  207,   83,   69, 1071,   21,
         21,   72,  567,  287,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0], dtype=int32)
```

[¶ TOP](#lstm-model)

# 데이터 세팅
## 학습용-평가용 데이터 분리

```py
from sklearn.model_selection import train_test_split

train_reviews, test_reviews, train_ratings, test_ratings = train_test_split(reviews, ratings, test_size = 0.2, random_state = 42)

print("training reviews shape:", train_reviews.shape)
print("testing reviews shape:", test_reviews.shape)
print("training ratings shape:", train_ratings.shape)
print("testing ratings shape:", test_ratings.shape)
#
Training reviews shape: (93275, 50)
Testing reviews shape: (23319, 50)
Training ratings shape: (93275,)
Testing ratings shape: (23319,)
```

## LabelEncoder

```py
from sklearn.preprocessing import LabelEncoder

# 레이블 인코딩: 범주형 데이터를 숫자로 변환하는 기법
label_encoder = LabelEncoder() 
y_train_encoded = label_encoder.fit_transform(train_ratings) # 학습용에 기준 맞추기
y_test_encoded = label_encoder.transform(test_ratings) # 평가용은 fit 제거
```

## 데이터셋 정의

```py
from torch.utils.data import DataLoader, Dataset

class ReviewDataset(Dataset):
  def __init__(self, reviews, ratings):
      self.reviews = reviews
      self.ratings = ratings
       
  def __len__(self):
      return len(self.reviews)
  def __getitem__(self, idx):
     review = self.reviews[idx]
     rating = self.ratings[idx]
     return torch.tensor(review), torch.tensor(rating)


BATCH_SIZE = 20
# 최초 64개 설정 
# 21 개 이상으로 넣어도 21개로 분할, 21 이하로 조정
# 2의배수가 학습에 용이, 20으로 수정

train_dataset = ReviewDataset(train_reviews, train_ratings)
test_dataset = ReviewDataset(test_reviews, test_ratings)
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, drop_last=True)
# 20개인것만 학습하기 위해 drop_last = True
```

## batch 생성 확인

```py
for reviews, ratings in train_dataloader:
        print(reviews.shape)
        print(ratings.shape)
```

# LSTM 모델 정의
- 예비용 파라미터, 손실함수 등의 옵션 참조처리

```py
import torch.nn as nn
import torch.optim as optim

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True) # sgd
        # self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False) # adam
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden[-1])

# 하이퍼파라미터 정의
VOCAB_SIZE = len(tokenizer.word_index) + 1 
# 단어집에 없는 단어는 <'OOV'> 처리 했기 떄문에 1 추가
EMBED_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = len(label_encoder.classes_)  # 예측할 클래스 개수 # 5 

# len(set(ratings)로 하면 21이 나옴 - 인코딩 되지 않아서 인지 tensor 중복값 허용

# 모델 초기화
model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)

#

LSTMModel(
  (embedding): Embedding(50734, 64, sparse=True)
  (lstm): LSTM(64, 128, batch_first=True)
  (fc): Linear(in_features=128, out_features=5, bias=True)
  )
```

## 학습 루프

```py
import time
start_time = time.time()
model.train()
num_epochs = 10
for epoch in range(num_epochs):
    for reviews, ratings in train_dataloader:
        optimizer.zero_grad()
        outputs = model(reviews)
        # 이전 단계에서 계산된 기울기 초기화
        loss = criterion(outputs, ratings)
        loss.backward()
        # 역전파를 통한 기울기 계산
        optimizer.step()
        # 가중치 업데이트

    if (epoch + 1) % 1 == 0: # 1회 마다 출력
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Finished Training')

end_time = time.time()

# 총 학습 시간 출력
print(f'Training time: {end_time - start_time:.2f} seconds')
print('Finished Training')
```

## 평가

```py
correct = 0
total = 0

model.eval()  # 평가 모드로 설정
with torch.no_grad():  # 기울기 계산 비활성화
    for data in test_dataloader:  # 테스트 데이터 로더에서 데이터 반복
        images, labels = data  # 이미지와 레이블 가져오기
        outputs = model(images)  # 모델을 통해 예측
        _, predicted = torch.max(outputs.data, 1)  # 가장 높은 확률을 가진 클래스 인덱스 선택
        
        total += labels.size(0)  # 총 샘플 수 누적
        correct += (predicted == labels).sum().item()  # 정확한 예측 개수 누적

# 정확도 출력
print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')
#
Accuracy of the network on the test images: 62.13%
```
[¶ TOP](#lstm-model)

# 예측 함수 - Pipeline

```py
def predict_review(model, review):
    # 리뷰 전처리 및 예측
    def text_pipeline(review):
        processed_review = preprocess_text(review)  # 기존의 전처리 함수 사용
        sequence = tokenizer.texts_to_sequences([processed_review])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, truncating='post', padding='post')
        return padded_sequence[0]  # 첫 번째 (유일한) 샘플 반환

    model.eval()
    with torch.no_grad():
        tensor_review = torch.tensor(text_pipeline(review), dtype=torch.long).unsqueeze(0)  # 배치 차원 추가
        # dtype = torch.long : int64 타입
        output = model(tensor_review)
        prediction = output.argmax(1).item() # 가장 큰 값
        return label_encoder.inverse_transform([prediction])[0]
        # 인코딩된 것을 정수형태로 복원, 리스트 형태 필요 [] 처리 , [0] 첫번째 것

# 새로운 리뷰에 대한 예측
new_review = "LSTM is very hard to use!!!!"
predicted_score = predict_review(model, new_review)
print(f'Score: {predicted_score}')
#
Predicted Score: 1
```

[¶ TOP](#lstm-model)

# 파라미터 조정

```py
# sgd 1 - MAX_LENGTH = 150
Epoch [1/10], Loss: 1.3247
Epoch [2/10], Loss: 1.4765
Epoch [3/10], Loss: 1.5148
Epoch [4/10], Loss: 1.4463
Epoch [5/10], Loss: 1.4844
Epoch [6/10], Loss: 1.4649
Epoch [7/10], Loss: 1.4446
Epoch [8/10], Loss: 1.5899
Epoch [9/10], Loss: 1.3331
Epoch [10/10], Loss: 1.3684
Finished Training
Training time: 2430.85 seconds
Finished Training

# sgd 2 - MAX_LENGTH = 50
Epoch [1/10], Loss: 1.3495
Epoch [2/10], Loss: 1.2912
Epoch [3/10], Loss: 1.2678
Epoch [4/10], Loss: 1.0818
Epoch [5/10], Loss: 1.0717
Epoch [6/10], Loss: 1.5727
Epoch [7/10], Loss: 0.9725
Epoch [8/10], Loss: 1.4701
Epoch [9/10], Loss: 1.1300
Epoch [10/10], Loss: 0.9992
Finished Training
Training time: 1359.31 seconds
Finished Training

# sgd 3 - MAX_LENGTH = 50, laber_encoder 적용: 시간 대폭 단축
Epoch [1/10], Loss: 1.4144
Epoch [2/10], Loss: 1.3831
Epoch [3/10], Loss: 1.5715
Epoch [4/10], Loss: 1.3693
Epoch [5/10], Loss: 1.2890
Epoch [6/10], Loss: 1.0397
Epoch [7/10], Loss: 0.8751
Epoch [8/10], Loss: 0.9583
Epoch [9/10], Loss: 1.1231
Epoch [10/10], Loss: 0.9737
Finished Training
Training time: 672.36 seconds
Finished Training

# sgd 4 - MAX_LENGTH = 50, laber_encoder, EMBED_DIM = 128, HIDDEN_DIM = 256
Epoch [1/10], Loss: 1.3284
Epoch [2/10], Loss: 1.3181
Epoch [3/10], Loss: 0.9882
Epoch [4/10], Loss: 0.8318
Epoch [5/10], Loss: 0.9391
Epoch [6/10], Loss: 0.6783
Epoch [7/10], Loss: 1.2156
Epoch [8/10], Loss: 0.8556
Epoch [9/10], Loss: 1.1602
Epoch [10/10], Loss: 0.9620
Finished Training
Training time: 1363.60 seconds
Finished Training
Accuracy: 63.42%

# sgd 5 - MAX_LENGTH = 50, laber_encoder, EMBED_DIM = 64, HIDDEN_DIM = 128

Epoch [1/10], Loss: 1.4150
Epoch [2/10], Loss: 1.2739
Epoch [3/10], Loss: 0.8576
Epoch [4/10], Loss: 1.2630
Epoch [5/10], Loss: 0.8323
Epoch [6/10], Loss: 0.8134
Epoch [7/10], Loss: 1.0209
Epoch [8/10], Loss: 0.9471
Epoch [9/10], Loss: 0.7644
Epoch [10/10], Loss: 0.7247
Finished Training
Training time: 798.79 seconds
Finished Training
Accuracy: 63.11%
```

[¶ TOP](#lstm-model)