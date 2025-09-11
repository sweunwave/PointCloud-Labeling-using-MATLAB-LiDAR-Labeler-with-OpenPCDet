# PointCloud Labeling using MATLAB LiDAR Labeler with OpenPCDet

본 문서는 **MATLAB LiDAR Labeler**와 **OpenPCDet**을 활용하여  
PointCloud 데이터에 Auto-Labeling을 적용하는 방법을 설명합니다.

---

## 1. MATLAB 실행 및 LiDAR Labeler 실행
![LiDAR Labeler 실행](imgs/image.png)

---

## 2. PointCloud 불러오기
![PointCloud Load](imgs/image-1.png)

---

## 3. PCD 파일 경로 지정
- PCD 파일이 위치한 폴더를 지정하고 불러옵니다.  
![PCD 경로 지정](imgs/image-4.png)

---

## 4. Label 불러오기
- `label.mat` 파일을 설정하여 레이블 정의를 불러옵니다.  
![Label 불러오기](imgs/image-5.png)

---

## 5. gTruth 구조 저장
- 아직 Label 정보는 없지만, `gTruth` 구조체를 저장합니다.  
- 이후 OpenPCDet을 활용하여 Auto-Labeling을 수행하고, 생성된 레이블 정보를 해당 구조체에 덮어쓸 예정입니다.  
![gTruth 저장](imgs/image-6.png)

---

## 6. gTruth 저장 확인
- `load` 함수를 이용해 `gTruth`가 정상적으로 저장되었는지 확인합니다.  
![gTruth 확인](imgs/image-7.png)  
![gTruth 확인](imgs/image-8.png)

---

## 7. OpenPCDet을 활용한 `label_array` 생성
- OpenPCDet을 이용하여 Auto-Labeling을 수행합니다.  
- Python 코드(`pcdet_auto_labeling.py`) 실행 필요.  
- 결과로 `label_array_from_python_2.mat` 파일 생성.  

---

## 8. Auto-Labeled 정보 추가
```matlab
>> load("my_gTruth.mat")
>> load('label_array_from_python_2.mat')
>> at = array2timetable(label_array, 'TimeStep', seconds(1.0), 'VariableNames',{'Vehicle','Pedestrian','Cyclist'});
>> at.Time.Format = 'mm:ss.SSSSS';
>> gTruth2 = groundTruthLidar(gTruth.DataSource, gTruth.LabelDefinitions, at);
>> save("labeling_result.mat", "gTruth2")
```

* 코드 설명
    * 기존 ```gTruth``` 구조체의 **DataSource** 및 **LabelDefinitions** 유지
    * Auto-Labeled 정보를 ```gTruth2``` 구조체에 추가
    * 최종 결과를 ```labeling_result.mat```으로 저장

## 9. Auto-Labeled 결과 확인
- MATLAB LiDAR Labeler 실행 후, 8번에서 저장한 labeling_result.mat 파일을 불러옵니다.
![alt text](imgs/image-10.png)
![alt text](imgs/image-11.png)

