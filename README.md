# music-generator-project
### 프로젝트 시작 계기
머신러닝/딥러닝 기술을 이용해 새로운 가치를 창출하는 서비스를 만들어보고 싶어서 이프로젝트를 진행하게 되었습니다.  
주제를 음악작곡으로 하게된 이유는 소규모 미디어가 많은 요즘 저작권없는 상업용 음악의 수요가 늘어났기 때문입니다 인공지능 작곡 서비스는 비용과 시간적인 측면에서 기존의 작곡시스템보다 유리하다는 장점이 있습니다

### 과정
유명한 피아니스트이자 작곡가인 프란츠리스트의 10곡을 학습데이터로 사용하였습니다 전처리된 학습데이터는 pikle파일로 만들어 입력에 사용됩니다. 모델은 시계열 데이터에 적합한 lstm을 사용했고 pikle file에서 100개의 시퀀스를 모델에 넣으면 모델을 거쳐 499개의 note가 만들어집니다.
서비스를 배포하기위해 flask를 사용해서 웹페이지를 구축하였고 서버는 aws ec2를 사용

### 최종적인 목표
작곡되는 음악의 완성도를 높이기위해 멀티트랙모델을 만들고 장르를 사용자가 선택할수 있도록 해볼 예정입니다.
그리고 박자, 장조, 세션과 같은 세부사항도 조율할 수 있도록 하고 싶습니다

