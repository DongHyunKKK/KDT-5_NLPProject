# KDT-5_NLPProject
경북대학교 KDT(Korea Digital Training) 자연어처리(NLP) 프로젝트

## 방언 분류 모델

  
#### DATA
- AI hub  
[[https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/amandam1/120-dog-breeds-breed-classification)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71517)
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71558
  
#### 역할분담 및 기능설계서

![image](https://github.com/KDT5-1TEAM/KDT-5_NLPProject/assets/155441547/64e09c2a-66d9-4994-b100-3985fbf78d0a)






<details>
  <summary>
    박희진( 경상도 방언 분류 모델 및 exe 파일 )
  </summary>

# 경상도 사투리 이진 분류 모델

## (1) 데이터 확인 및 전처리

- Josn 형식인 원본 데이터를 csv형식으로 바꿔  pd.read_csv로 불러옴
    - 사투리와 사투리의 표준어 버전 정보 두개를 들고옴
- 전체 데이터 개수 : 212906개
- 중복치 개수 : 117795개
    - 여기서 중복치란, 사투리와 표준어의 차이가 하나도 없어 중복치로 간주되는 경우
    - 따라서 사투리 데이터는 100000개, 표준어 데이터는 200000개라는 뜻이 됨 → 심한 데이터 불균형
    - 중복치를 삭제할까 고민했지만 표준어 데이터로서 학습시키기 위해 제거하지 않았음
        - 다운샘플링을 했을 때, 모델 성능 저하와 데이터 소실 고려
- melt를 이용해 컬럼명을 하나의 컬럼으로 만들어 줌
    - 사투리 / 표준어
- string에 있는 punctuation을 이용해 각 문장에 존재하는 구두점을 제거
- 불용어를 제거할 지 말 지 고민했지만, 사투리에 있어서 표준어와 구분되는 의미있는 부분은 우리가 불용어로 간주하는 조사와 대명사 등에 많이 존재하기 때문에 불용어 제거

## (2) 데이터셋 준비

- 데이터셋 클래스 내에서 value 컬럼이 사투리라면 1, 아니라면(표준어라면) 0으로 라벨링 되게끔 설정
- 텍스트 데이터와 라벨 데이터가 넘파이 배열일 경우와 판다스데이터프레임일 경우 둘 다 고려해서 객체 인스턴스 생성 함수 만듦
    - 그 외 함수는 가장 기본적인 것만
- 라벨의 비율을 균형적으로 맞춰 학습시켜주기 위해 sklearn의 train_test_split 사용
    - train_test_split 사용하기 위해 불러온 데이터셋에서 텍스트와 라벨을 각각 다른 리스트 내에 분리
    - 학습용 데이터와 검증용 데이터는 8 : 2로 분리
- 개수를 확인해보니 균형적이게 잘 들어갔음
    - 그러나 사실 표준어가 더 많은 불균형 데이터임을 후에 인지함

## (3) 어휘사전 생성

- 형태소 분석기로는 Mecab 사용
- 경상도 사투리는 종결어미에서 그 특징이 두드러지기 때문에 어간 추출하지 않음
    - 어미 제거도 하지 않음
- 사투리 어휘사전을 만들어야하기 때문에 당연히 표제어 추출도 하지 않음
- build_vocab_iterator를 통해 어휘사전 생성
- 후에 예측할 때, 모델 별로 적용되는 어휘사전이 달라야하기 때문에 경상도 어휘사전을 피클 파일로 저장

## (4) 데이터 로더 생성

- 패딩을 하지 않고, offset 정보를 가지고 문장을 구별해줄 것이기 때문에 collate_batch() 함수 생성
    - 라벨과 텍스트, 오프셋 정보를 반환하도록 구현
- 배치사이즈는 16384로 지정
    - 후에 모델을 학습했을 때, 과대적합이 너무 심하게 일어나서 조정한 값
    - 10000 이상인 2의 배수 선택
    - collate_fn = collate_batch

## (5) Custom Text 모델 클래스

- RNN 모델로는 GRU 사용
    - 문장이 긴 해당 데이터에 적합하다고 판단
    - 시간도 한정되어 있기 때문에 최대한 효율적인 모델 선택
    - 사투리 문장에서 앞뒤 문맥 파악도 중요하기 때문에 양방향으로 설정
- RNN 모델 층 수는 1개로 고정
    - 혹시나 해서 2개로 늘여봤더니, 모델 성능도 저하되고 과대적합도 심해짐
    - 셀의 개수는 3개로 설정
- 객체 생성할 때, 가중치 초기화 설정
    - 이때까지도, 균형 데이터라고 생각하고 uniform으로 가중치 초기화
- offset값을 이용할 것이기 때문에 embeddingBag 사용
- drop out  층 추가
    - drop out되는 비율은 0.2로 설정

## (6) 학습준비 - 학습함수, 평가함수

- 에포크는 30으로 지정
    - 학습 함수에 스케쥴러를 이용해 조기 종료 기능 구현
    - Valid Loss가 3번 이상 개선이 안되면 조기 종료
- 이진 분류 모델이기 때문에 BCEWithLoss 손실함수 이용
    - 해당 손실 함수는 시그모이드를 내장하고 있기 때문에, 모델 내에 시그모이드 활성화 함수를 넣어줄 필요 X
- 옵티마이저는 Adam 이용
    - 러닝메이트는 0.01로 지정

## (7) Custom RNN 모델 평가

![image](https://github.com/ParkHeeJin00/KDT-5_NLPProject/assets/155441547/605fcb1d-f1ce-4528-8555-de22448d1e85)


- 처음에 정의했던 RNN 클래스의 모델은 과대적합이 매우 심했음
    - 과대적합을 줄이기 위해서 여러 시도를 함
        - 배치사이즈를 늘려보기( 위 모델의 BATCH_SIZE는 64 )
        - drop out을 모델 층에 넣기
        - 제거했던 중복치를 되돌리기
        - 모델 복잡도 줄이기
- BATCH_SIZE가 10000이상은 되었을 때, loss 감소 추세라던지 score 증가 추세가 안정적인 모양을 띠게 됨
- 최종 모델
    - Batch size = 16384
    - using train_test_split
    - drop_out(0.2)
    - rnn bidirectional=True
    
    > Train Loss : 0.3444
    Train F1 Score : 864882
    > 
    
    > Valid Loss : 0.3569
    Valid F1 Score : 842564
    > 
- 최종 모델에서 과대적합이 어느 정도 해결된 것을 볼 수 있음

## (8) 예측 및 느낀 점

 predict를 해봤을 때, 내 데이터가 심한 불균형 데이터라는 것을 알게 되었다. 따라서 임계치를 데이터가 많은 쪽으로 이동시켜 데이터 불균형을 어느 정도 해결했다. 임계치를 이동했더니 모델 성능이 높아졌다.
</details>
  
<details>
  <summary>
    김동현( 전라도 방언 분류 모델 )
  </summary>
  
</details>
  
<details>
  <summary>
    김문섭( 제주도 방언 분류 모델 및 5도 사투리 분류 모델 )
  </summary>

</details>
