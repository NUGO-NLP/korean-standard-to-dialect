NUGO Project
=====================

표준어 to 사투리 변환기

## 1. Run

```
$ pip3 install -r requirements.txt
$ python3 main.py --dialect gs --input_level syl
```

--dialect : Target dialect (default='gs')
--input_level : Input level of train data. Supported input levels are [syl, word, jaso] (default='syl')
--maxlen : Max length of target data (default=110 if input_level=='syl',
                                     default=30 if input_level=='word')
--train : Adding this will train model

## 2. Result

```
[표준어 Input]  글쎄요 저는 추천해드리고 싶지 않습니다
[경상도 Output] 글쎄요 저는 추츤해드리고 싶지 않심더
[전라도 Output] 글씨요 저는 추천해드리고 싶지 않아요
```
```
[표준어 Input]  좋아하는 영화는 어떤 장르입니까
[경상도 Output] 좋아하는 영하는 어떤 장르입니꺼
[전라도 Output] 좋아하는 영화는 우떤 장르입니까
```
```
[표준어 Input]  여기 사고 신고서를 작성해 주세요
[경상도 Output] 여기 사고 신고스를 작승해 주세여
[전라도 Output] 여그 사고 신고서를 작성해 주세요
```
```
[표준어 Input]  키즈룸은 지하 일층에 있습니다 키즈룸을 이용해 보시겠습니까
[경상도 Output] 키즈룸은 지하 일층에 있심더 키즈룸을 이용해 보시겠심꺼
[전라도 Output] 키즈룸은 지하 일층에 있당께요 키즈룸을 이용해 보시겠습니까
```
```
[표준어 Input]  죄송합니다 이 제품이 한 번도 이런 적이 없었는데요 고장 접수해드리겠습니다
[경상도 Output] 죄송합니데이 제품이 한 번도 이런 즉이 없었는데예 고장 즙수해드리겠심더
[전라도 Output] 죄송헙니다 이 제품이 한 번도 이런 적이 없었는디요 고장 접수해드리겠습니다
```
```
[표준어 Input]  적발되면 중절도죄와 일급 문서위조죄가 적용돼 정식 재판에 회부되며 징역형까지 선고될 수 있어요
[경상도 Output] 즉발되면 중절도죄와 일급 문서위조죄가 즉용돼 정식 재판에 회부되며 징역형까지 슨고될 수 있어예
[전라도 Output] 적발되면 중절도죄와 일급 문서위조죄가 적용돼 정식 재판에 회부되며 징역형까지 선고될 수 있으요
```

