# gestureImageViewer

- 파이썬 개발
- 단일 이미지 뷰어
- mediapipe의 hand landmark 라이브러리를 통해 라이브 카메라로부터 양 손 조인트 위치 받아오기
- 조인트 위치 사용해서 제스처 판단
    - 제스처: 한 손 핀치, 양 손 핀치
    - 액션: 
        - 이미지 확대: 두 손 엄지, 검지 붙였다가 때면 발생하는 직사각형 크기에 비례해서 이미지 확대
        - 이미지 틸트: 한 손 엄지, 검지, 중지 붙였다가 때고 회전 시 발생하는 원과 기준선 회전에 비례하여 이미지 회전
        - 이미지 확대된 상태에서 움직이기: 확대된 상태에서 한 손 핀치 후 핀치 이동시키면 핀치 위치에 따라 이미지 이동

To-do:  
1. mediapipe hand landmark python으로 써보기
2. image viewer 만들기
3. 알고리즘 구현  

알고리즘:
- mediapipe 반환값에서 필요한 것: 엄지(index 4), 검지(index 8), 중지(index 12) tip 상대위치좌표 (x,y,z)
- 제스처 인식 알고리즘 input: tip 위치좌표, output: 제스처 판정
- 액션 알고리즘 input: tip 위치좌표, 제스처 판정, output: 액션 apply중인 이미지