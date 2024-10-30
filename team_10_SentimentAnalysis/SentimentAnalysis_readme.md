# Sentiment Analysis
>자연어 감성 분석<br>
[¶ Preprocessing](#데이터-전처리)<br>
[¶ Sentiment Analysis](#감성분석)<br>
[¶ Add Result](#감성분석-결과-데이터-추가)<br>
[¶ WordCloud](#wordcloud)

# 데이터 전처리 
## 불러오기, 결측치 처리

```py
import pandas as pd

df = pd.read_csv("/netflix_reviews.csv")

df['content'] = df['content'].fillna("so so")
df.duplicated(subset=['reviewId', 'userName', 'content', 'score']).sum() # 337
df = df.drop_duplicates(subset=['reviewId', 'userName', 'content', 'score'], keep = 'first')
# 첫번째만 남김

df.duplicated(subset=['reviewId', 'userName', 'content', 'score']).sum() # 0
```

## 학습에 필요한 데이터만 분리

```py
df['content']
content = df['content']
socre = df['score']
```

## 데이터 개요 파악
- `RegexpTokenizer` : 정규표현식 조건을 적용한 뒤 토큰화
- `Text` : **NLTK** 모듈 클래스 선언, 통계 기능 사용 가능
- 결과: 해석에 불필요한 단어 다수 포함

```py
import nltk
from nltk import Text
from nltk.tokenize import RegexpTokenizer

retokenize = RegexpTokenizer(r"[\w]+") # 문자만

content_1st_lot = ' '.join(content) # 전체 문장을 하나로 통합
lot_1 = Text(retokenize.tokenize(content_1st_lot)) # 토큰화 일괄 적용
lot_1.plot(30) # 상위 30개의 단어 출력
```
![](/LLM/images/lot_1.png)

## 불용어 설정
- `from nltk.corpus import stopwords` : 불용어 사전
- `.union(list)` : 불용어 사전에 사용자 정의 단어 추가

```py 
stop_words = set(stopwords.words('english'))
additional_stopwords = {'app', 'netflix', 'show', 'time', 'series', 'phone', 'movie', 'tv', 'would', 'watch'}
stop_words = stop_words.union(additional_stopwords)
stop_words.discard('not')
# not 이 빠지면 의미가 달리지는 경우가 있어 불용어에서 제외
```

## 전처리 함수 설정

### 표제어 추출 
`from nltk.stem import WordNetLemmatizer` 
- 동사 원형 추출 : 동사의 형태 통일
- 명사 원형 추출 : 명사의 형태 통일

```py
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def lemma_v(sentence): # 동사
    return [lemmatizer.lemmatize(word, 'v') for word in sentence] 

def lemma_n(sentence): # 명사
    return [lemmatizer.lemmatize(word, 'n') for word in sentence]
```

### 불필요한 품사[¶](#pos-태그-목록) 제거 
`from nltk.tag import pos_tag` : 태그 부착 후 태그를 기준으로 정리
- 고유명사, 인칭 대명사, 관사 제거

```py
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def TagFitter(sentense):
  tokens = pos_tag(word_tokenize(sentense)) # 토큰화
  tags = [t[0] for t in tokens if t[1] != "NNP" and t[1] != "NNPS" and t[1] != "PRP" and t[1] != "DT"]
  # 특정 품사에 해당하지 않는 단어추출
  return ' '.join(tags) # 문장으로 복원
```

### 단어 대체
- `'t` 는 토큰화 하면 의미 없는`t`로 출력됨
- `not`으로 대체 

```py
def t_replacer(sentense):
  tokens = word_tokenize(sentense)
  r = []
  for token in tokens:
    if token == 't': 
      r.append('not')
    else: 
      r.append(token)
  return ' '.join(r)
```

### 철자 검토 함수 
- `from textblob import TextBlob` : 철자 검토 기능 
- `.correct()`

```py
from textblob import TextBlob
def WordCorrect(senntense):
  s = TextBlob(senntense)
  s = s.correct()
  s = ''.join(s)
  return s
```

### 종합하여 전처리 함수 작성
- 주의점
  - 고유명사 태그는 대문자를 기준으로 하는 경우가 많기 떄문에 `.lower()` 전에 작성
  - 표제어 추출을 먼저 작성하여 원형만 사용 : 불용어 사전 최소화
  - 연산 효율화, 철자검사 및 불용어 처리를 위해 소문자화
  - 철자 검사는 처리 결과를 확인하고 사용할지 선택

```py
def preprocessing(sentence):
  if isinstance(sentence, float): return '' # 실수형 데이터 제거, 문자형만
  cleaned = re.sub('[^a-zA-Z]', ' ', sentence) # 문자만
  cleaned = t_replacer(cleaned) # 't 를 not으로 대체
  cleaned = TagFitter(cleaned) # 태그를 기준으로 불필요한 품사 제ㄱ
  cleaned = cleaned.lower() # 소문자화
  cleaned = cleaned.strip() # 띄어쓰기 제외한 공백 제거
  cleaned = cleaned.split() # 문장 분할
  cleaned = lemma_v(cleaned) # 동사 원형화
  cleaned = lemma_n(cleaned) # 명사 원형화
  cleaned = [word for word in cleaned if word not in stop_words] 
  # 불용어 제거
  cleaned = ' '.join(cleaned) # 문장으로 복원
  # cleaned = WordCorrect(cleaned) # 철자 검사
  return cleaned

content[0:5].apply(preprocessing)
  # 
0                                             not open
1                                                 best
2    famous korean drama not dub sense pay subscrip...
3       superb please add comment section like youtube
4    reason not give four star opinion many foreign...
Name: content, dtype: object
```

## 전처리 결과 확인
- **not**은 추후에 제거
- 혼동을 피하기 위해 데이터 명명에 주의

```py
content_2nd_lot = content.apply(preprocessing)
df['content_c'] = content.apply(preprocessing)
content_2nd_lot = ' '.join(content_2nd_lot)

lot_2 = Text(word_tokenize(content_2nd_lot))
lot_2.plot(30)
```

![](/LLM/images/lot_2.png)


[¶ Top](#sentiment-analysis)

# 감성분석
## Sentiment
- `TextBlob`, `.sentiment`
- 감성분석의 긍정/부정 수치를 출력하는 함수 작성
- **-1.0 ~ 1.0**

```py
def sentiment(sentense):
 senti = TextBlob(sentense).sentiment
 polar = senti.polarity
 return polar

test_lot = content[0:10].apply(preprocessing)
test1 = test_lot.apply(sentiment)
test1.head(10)
#
0    0.000000
1    1.000000
2    0.350000
3    1.000000
4    0.036667
5    0.000000
6    0.214286
7    0.700000
8    0.000000
9    0.000000
Name: content, dtype: float64
```

## NaiveBayes
- **NaiveBayes** 모델 적용
- 학습에 시간이 걸림

```py
from textblob.sentiments import NaiveBayesAnalyzer
def NB_cl(sentense):
  senti = TextBlob(sentense, analyzer = NaiveBayesAnalyzer())
  cl = senti.sentiment.classification
  return cl

test2 = test_lot.apply(NB_cl)
test2.head(10)
```

# 감성분석 결과 데이터 추가

```py
df['sentiment'] = (content.apply(preprocessing)).apply(sentiment)
df['sentiment']
#
0         0.000000
1         1.000000
2         0.350000
3         1.000000
4         0.036667
            ...   
116926    0.350000
116927    0.450000
116928    0.050000
116929   -0.500000
116930    0.500000
Name: sentiment, Length: 116594, dtype: float64
```

## 결과에 라벨 부여

```py
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))
df[['content_c', 'score', 'sentiment_label']]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content_c</th>
      <th>score</th>
      <th>sentiment_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>not open</td>
      <td>1</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>best</td>
      <td>5</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>famous korean drama not dub sense pay subscrip...</td>
      <td>2</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>superb please add comment section like youtube</td>
      <td>5</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>reason not give four star opinion many foreign...</td>
      <td>1</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>116926</th>
      <td>really like many kdramas</td>
      <td>5</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>116927</th>
      <td>love always enjoy use</td>
      <td>5</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>116928</th>
      <td>sound quality slow</td>
      <td>1</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>116929</th>
      <td>rate expensive bcos see sunday charge hole month</td>
      <td>1</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>116930</th>
      <td>awesome english</td>
      <td>4</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
<p>116594 rows × 3 columns</p>
</div>

[¶ TOP](#sentiment-analysis)

# WordCloud 
- 리뷰 감성별 주요 단어 추출

## 데이터 전처리
### 감성별로 데이터 분리

```py
pos = df[df['sentiment_label'] == 'positive']
neu = df[df['sentiment_label'] == 'neutral']
neg = df[df['sentiment_label'] == 'negative']

positive = pos['content_c']
neutral = neu['content_c']
negative = neg['content_c']
```

## Not 제거
- 워드클라우드에 **not**이 포함되지 않게 제거

```py
rm_not = ['not']

def remove_not(sentense):
  cleaned = sentense.split()
  cleaned = [word for word in cleaned if word not in rm_not]
  cleaned = ' '.join(cleaned)
  return cleaned

positive = positive.apply(remove_not)
neutral = neutral.apply(remove_not)
negative = negative.apply(remove_not)
```

# Gensim - LDA 모델 적용
> **Latent Dirichlet Allocation**<br>
잠재 디리클레 할당: 문서집합에서 토픽을 찾아내는 프로세스

## LDA 모델 작성

```py
from gensim.models import LdaModel
def LDA_model(serise):
  preprocessed = serise.apply(simple_preprocess) # 전처리
  dictionary = corpora.Dictionary(preprocessed) # 코퍼스 생성
  bow = [dictionary.doc2bow(doc) for doc in preprocessed] # Bag of Word ; 단어집 생성
  return LdaModel(bow, num_topics=1, id2word=dictionary, passes=10) # passes : 반복 횟수

pos_model = LDA_model(positive)
neu_model = LDA_model(neutral)
neg_model = LDA_model(negative)
```

## 토픽 추출
`.show_topic(idx)`

```py
pos_model.show_topic(0)
# 단어, 비율
[('good', 0.036281966),
 ('love', 0.03461763),
 ('great', 0.018250687),
 ('like', 0.01683506),
 ('best', 0.015260779),
 ('get', 0.011255411),
 ('really', 0.011232897),
 ('use', 0.01018283),
 ('new', 0.008160302),
 ('work', 0.0079550715)]
```

## WordCloud 생성
- 단어별 빈도수를 계산하여 출력

```py
for idx in range(pos_model.num_topics):
    word_freq = dict(pos_model.show_topic(idx, topn=200))  
    # 인덱스 를 사용하여 단어와 빈도수 가져오기
    # topn : 가져올 단어 수
    
    # 워드클라우드 생성
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(word_freq) # 빈도수

    # 워드클라우드 시각화
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear') # 보간법 = 선형
    plt.axis('off')  # 축 제거
    plt.title('Positive Reviews Word Cloud')
    plt.show()
```

![](/team_10_SentimentAnalysis/images/pos_wc.png)

### 토픽 추출 없이 바로 생성

```py
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# 긍정적 리뷰 필터링 : #5번에서 분류한 'sentiment_label' 열에서 값이 'positive' 인 행 선택-> 필터링 된 df 에서 'content' 열 선택해 긍정적 리뷰 텍스트 추출
positive_reviews = ' '.join(df[df['sentiment_label'] =="positive"]['content'])

# 불용어 설정
stopwords = set(STOPWORDS)
stopwords.update(['watch','account','netflix','use','im','new','will','shows','video','screen','update','now','device', 'movie', 'show', 'time', 'app', 'series', 'phone','movies'])
# update() 메서드 : 주로 set 과 dict 형태에 사용됨. 주어진 리스트,튜플,집합 등의 요소를 현재 집합에 추가. 중복 요소 무시

# WordCloud 생성
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(positive_reviews)

# WordCloud 출력
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off') - 궁금해서 축 활성화 
plt.title('Positive Reviews Word Cloud')
plt.show()
```

![](/team_10_SentimentAnalysis/images/pow_wc_2.png)

### 마스크 이미지 활용

```py
# 마스크 이미지 사용
# 부정적인 리뷰 필터링
negative_reviews = ' '.join(df[df['sentiment_label'] =="negative"]['content'])

import numpy as np
from PIL import Image
# 마스크 이미지를 사용하는 법 
mask = np.array(Image.open('/Users/users/myenv/cloud.png'))

# 불용어 설정
stopwords = set(STOPWORDS)
stopwords.update(['watch','account','netflix','use','im','new','will','shows','video','screen','update','now','device', 'movie', 'show', 'time', 'app', 'series', 'phone','movies'])
# update() 메서드 : 주로 set 과 dict 형태에 사용됨. 주어진 리스트,튜플,집합 등의 요소를 현재 집합에 추가. 중복 요소 무시



# WordCloud 생성 : WordCloud(mask=마스크이미지,contour_color=윤곽선색상,contour_width=윤곽선두께,width=가로크기,height=세로크기,max_words=최대단어수,background_color=배경색,colormap=단어색상)
wordcloud = WordCloud(mask=mask,contour_color='black',contour_width=1, width=800, height=400, background_color='white', stopwords=stopwords).generate(negative_reviews) # generate() : 주어진 텍스트 데이터를 기반으로 워드 클라우드를 만듬

# WordCloud 출력
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') # 축 숨기기
plt.title('Negative Reviews Word Cloud')
plt.show()
```

![](/team_10_SentimentAnalysis/images/mask_wc.png)

[¶ TOP](#sentiment-analysis)