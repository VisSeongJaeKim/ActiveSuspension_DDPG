# ActiveSuspension_DDPG

1. 프로젝트 개요
  - 프로젝트 명 : DDPG기반의 차체 진동제어기
  - 연구 목적 : 차체의 진폭 감소를 하기 위한 차체의 높이 제어

2. 사용한 데이터셋
  - 1/4 차량 서스펜션 모델에서 나오는 변위 및 속도
  
3. 사용한 모델 : Deep Deterministic Policy Gradient

4. 딥러닝 라이브러리 : torch

5. 실험 및 결과 요약

Result 1
<br/>![사진1](https://github.com/VisSeongJaeKim/ActiveSuspension_DDPG/blob/master/images/Rewardpg.PNG)
  - Reinforcement Learning의 학습되는 것을 알아보는 척도로 Reward를 사용할 수 있는데 이 때 Reward가 0에 수렴하는 것으로 보아 
    잘되는 것을 알 수 있다.

Result 2
<br/>![사진2](https://github.com/VisSeongJaeKim/ActiveSuspension_DDPG/blob/master/images/cospg.PNG)

Result 3
<br/>![사진3](https://github.com/VisSeongJaeKim/ActiveSuspension_DDPG/blob/master/images/steppg.PNG)
  - 노면의 상태가 바뀌어도 DDPG를 이용한 Active Suspension 사용시 차체의 변위가 줄어든 것을 확인할 수 있다.

6. 코드에 대한 간략 설명
  - state_space_model.py
    강화학습에 사용될 environment에 해당됩니다. 모델의 변수, reward식, 각episode 당 끝나는 조건을 바꿀 수 있습니다.
  - main.py
    environment를 불러오고, episode를 몇번 실행시킬건지 결정하고, actor-critic network를 학습시키고 테스트를 합니다.
