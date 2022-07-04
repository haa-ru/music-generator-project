# music-generator-project
## 프로젝트 기획 배경

`머신러닝/딥러닝 기술을 이용하여 새로운 가치를 창출하는 서비스`를 만들고 싶어서 이 프로젝트를 진행하게 되었습니다. 주제선정 계기는 `소규모 미디어가 많아진 요즘, 개인의 상업용 음악에 대한 수요가 늘어났기 때문`이며 인공지능 작곡서비스를 만든다면 다음과 같은 이점을 기대 할 수 있을 것 입니다.

1. **무한히 많은 곡을 만들어 낼 수 있지만 그에 따른 작곡비용이 추가되지 않기 때문에 비용적인 측면에서 유리합니다**
2. **이미 만들어진 모델을 사용해 굉장히 빠른 시간안에 작곡이 가능합니다.**
3. **시간과 비용의 부담이 없어 좋은 결과물이 나올때까지 계속해서 새로운 곡을 만들어 낼 수 있습니다.**
    1. 많은 노력을 들인 창작의 결과물이 언제나 만족스러운 것은 아닙니다. 때문에 작곡가들은 `좋은 결과물이 나올때까지 작곡을 반복하는 방식`으로 문제를 해결해왔습니다. AI작곡은 시간과 비용이 많이 드는 `이러한 창작과정을 매우 빠르게 반복`할 수 있습니다. 
4. **수요자와 작곡자가 일치하기 때문에 세부적인 조절이 쉽고 만족도가 높습니다.**
    1. 기존의 작곡 시스템은 수요자와 작곡가가 별개로 존재하며 작곡가의 위주로 작곡이 진행되었습니다. 그에 비해 인공지능 작곡에서는 수요자가 곧 작곡가라고 볼 수 있습니다. 때문에 서비스에서 지원하는 세부조절 기능에 따라 다르겠지만 만드는 사람이 원하는만큼의 세부조절과 변형이 가능합니다.

물론 창작이라는 특성상 개별 곡의 완성도는 기존 작곡가들의 음악이 더 우수할 것입니다. 그러나 예술적창작이 아닌 실용성에 초점을 둔 개인에게는 위와 같은 인공지능작곡 서비스가 더 적합할 것 같습니다.

## 사용 데이터

학습데이터로는 유명한 피아니스트이자 작곡가인 프란츠리스트의 10곡을 사용하였습니다. 

리스트의 곡은 빠르고 화려하다는 특징을 가지고 있는 만큼 학습할 수 있는 음의 밀도가 높고, 비교적 작곡가의 느낌이나 특징이 잘 드러나 있어 학습에 적절하다고 생각했습니다.

> [Lacrimosa](https://www.youtube.com/watch?v=rKl4B75td70)
[Schubert/Liszt - Auf dem Wasser zu singen](https://www.youtube.com/watch?v=hrOxzR5VvYk)
[Mazeppa](https://www.youtube.com/watch?v=K9BQ1ylApto)
[Transcendental Étude No. 10 in F minor](https://www.youtube.com/watch?v=uDbZ-AcgmDE)
[Mephisto Waltz No. 1](https://www.youtube.com/watch?v=6fiDT1ZkdYo)
[Hungarian Rhapsody No. 2](https://www.youtube.com/watch?v=wkNccP146Hk)
[Paganini Liszt Etude No. 6](https://www.youtube.com/watch?v=7Blf8Y527DY)
[Consolation No. 3 in D Flat Major S 172](https://www.youtube.com/watch?v=CS58YQaVIaA)
[BeethovenLiszt 5th Symphony](https://www.youtube.com/watch?v=ANTk-mX-G4Q)
[Liebestraum(dream of love)](https://www.youtube.com/watch?v=5sVNk-fSKRQ)
> 

## 제작 과정, 사용기술

### 학습 데이터 전처리

- mp3와 같은 소리파일을 사용하면 잡음제거등의 추가적인 전처리과정이 필요하다 생각해 악보데이터인 midi파일을 사용하였습니다
- 데이터는 Note(음계)와 Chord(코드,화음)으로 이루어져 있습니다. Note는 Pitch(음높이)와 Octave(옥타브)로 나누어 저장한 뒤 각각의 모든 종류를 뽑아내어 정수에 매핑하는 딕셔너리를 생성합니다
- 딕셔너리를 이용해 정수형태로 바꿔준 데이터를 입력시퀀스로 만들고 LSTM레이어에 맞게 입력형태를 변경한뒤 정규화시켜줍니다.
- 각 시퀀스의 길이는 100노트/코드(chord)이며 네트워크에서 다음 노트를 예측하기 위해서 도움이 되는 이전 100개의 노트를 사용합니다.

### 모델학습

- **LSTM 레이어**
어떤 시퀀스를 입력으로 넣었을 때 출력으로 또 다른 시퀀스 또는 행렬을 주는 순환신경망입니다.
- **dropout 레이어** 
모델을 학습시킬 때 오버피팅(overfitting)이 되는 것을 방지하는 방법입니다. 모든 뉴런으로 학습하는 것이 아니라 무작위로 학습을 쓸 뉴런을 정해서 학습을 진행하는 것입니다. mini-batch 마다 랜덤하게 되는 뉴런이 달라지기 때문에 다양한 모델을 쓰는듯한 효과를 냅니다.
- **Dense 레이어**
이전 레이어의 모든 뉴런과 결합된 형태의 레이어
- **Activation 레이어**
신경망이 노드의 출력을 계산하는 데 사용할 활성화 기능을 결정합니다.

### 음악생성과정

- 매번 다른 시퀀스를 입력으로 준다면, 아무것도 하지 않고도 매번 다른 결과를 얻을 수 있기 때문에 랜덤 함수를 이용합니다.
- 모델을 이용하여 500개의 음을 만들어줍니다. 500개의 음으로 만들어진 음악은 길이가 2분정도 됩니다.
- 출력된 데이터가 코드인 경우는 문자열을 여러 노트로 분할한뒤 각 노트의 문자열을 반복해 쌓아줍니다.
- 노트/코드 가 쌓일때마다 오프셋이 0.5씩 증가하면서 리스트를 만듭니다

### 웹페이지 생성

- flask를 사용해 웹페이지를 생성합니다
- make버튼을 눌렀을때 app.py에서는 output페이지를 여는 함수를 실행하게 됩니다
- output템플릿을 보내는 함수 안에 모델로 곡을 만든뒤 저장하는 코드를 만들어놓습니다
- 작곡이 완려되면 output템플릿으로 넘어가게 됩니다
- 서버는 aws ec2 리눅스 서버를 사용하였습니다.

## 어려움 / 해결 방법

### **같은 음만을 반복하는 곡들을 만들어내는 문제**

처음 학습된 모델로 작곡을 실행한 결과, `곡의 시작부터 끝까지 같은 음만을 반복하는 곡들을 만들어내는 문제`가 나타났습니다. 간간히 정상적인 곡이 만들어졌다는 것과 학습수를 높여도 같은 문제가 생겼기 때문에 모델의 문제는 아니라고 판단하였고 학습에 사용되었던 곡을 다시 살펴보게 되었습니다. 

그 결과 리스트 곡의 특징인 반복 때문이라는 생각이 들었습니다. 우리가 음악을 감상할 때에는 강한 음과 멜로디 위주로 듣습니다. 리스트의 곡도 마찬가지로 전체적으로 들었을때에는 조화로운 음악이지만 특유의 화려함을 만들어주는 음들의 반복이 모델 학습에 걸림돌이 되었습니다.

다음 곡은 슈베르트의 가곡인 마왕을 리스트가 편곡한 곡입니다. 처음 학습시 넣었던 곡 중 하나로 같은 음이 유독 반복되는 곡입니다.

[ Schubert/Liszt - Erlkönig ]

[https://www.youtube.com/watch?v=icDGuppXoi0](https://www.youtube.com/watch?v=icDGuppXoi0)

이후 `반복되는 음이 많은 곡을 제외`시키고 조금 더 다양한 스타일의 곡들을 학습 데이터로 만들었습니다 그 결과 같은 음이 반복되는 곡이 만들어지는 문제를 해결 할 수 있었습니다.

## 구현 결과

### **구현한 딥러닝 모델로 작곡된 곡**

[848.mp3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0e206766-0209-4f39-8a7e-f6317e39624c/848.mp3)

[53.mp3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/501039fb-22ca-47f9-838b-0f5bc5cef146/53.mp3)

[307.mp3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a9b25297-4c54-43c3-a486-19b9d170dcea/307.mp3)

[n.mp3](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a436ac4e-31a0-4d12-ba49-0736ee779054/n.mp3)

### **웹페이지 화면**

[ index.html ]

![1.png](/Users/haru/dev/mg_/img/1.png)

[ output.html ]

![2.png](/Users/haru/dev/mg_/img/2.png)

## 한계점 / 개선방향

`[ 버튼클릭(작곡)시 로딩이 오래걸리며 진행을 사용자가 확인할 수 있는 부분이 없다 ]`

flask로 웹서비스를 구축하였고 html템플릿을 가져와 웹으로 전송하는 구조입니다 작곡하는버튼을 눌렀을때 모델작곡이 진행되고 그 이후 output템플릿으로 변경되기 때문에 로딩중을 표시할수가 없었습니다. 이 부분을 해결하기 위해 output템플릿을 두가지로 나누어서 로딩중을 보여주는 템플릿을 먼저 띄우고 모델작곡을 진행한 뒤에 완료를 보여주는 템플릿을 띄우면 될것 같습니다.

`[ 음이 반복되는 구간이 많고 기승전결이 없이 밋밋하다 ]`

모델링에 리스트(피아니스트)곡을 사용하였는데 리스트곡의 특징인 빠른 연타가 모델링을 할때 반복되는 구간을 많이 학습하게 되었던것 같습니다. 리스트곡에서는 빠른연타가 있어도 적절한 멜로디와 구성이 있기 때문에 전체적인 곡의 흐름이 부드럽고 멜로디가 잘 들리지만 프로젝트에서 사용된 모델은 곡의 전체적인 구성과 흐름을 학습하지 못했기 때문에 이런 문제가 나타난것 같습니다. 

그 이유는 노트의 지속시간과 노트 간의 [오프셋](https://ko.wikipedia.org/wiki/오프셋_(컴퓨터_과학))을 다양하게 지원하지 않았기 때문이라고 생각하며 보안을 위해서는 더 많은 클래스를 추가하여 LSTM의 깊이가 더 깊어져야 할것 같습니다.

## 진행중인 부분

- 사이트를 배포중 일정 시간이 지나면 서버가 다운되는 문제를 해결하는 중입니다.

## 추후 발전 과제

- 작곡되는 음악의 완성도를 높이기위해 멀티트랙모델을 만들고 장르를 사용자가 선택할수 있도록 해볼 예정입니다. 그러기 위해선 장르별로 각각의 모델을 학습하거나 장르를 구분짓는 특성들을 학습해서 선택된 장르에 맞게 조절할 수 있게 하면 될것 같습니다. 두번째 방법은 하나의 곡에서 장르의 비율을 조절할 수 있다는 장점도 있을것 같습니다.

- 박자, 장조, 세션과 같은 세부사항을 조율하는 기능도 구현한다면 더 좋은 서비스가 될 것 같습니다. 그러기 위해선 세부사항을 조절하는 기능과 함께 변경사항을 바로 모니터링 할 수 있는 기능도 필요할 것 같습니다.