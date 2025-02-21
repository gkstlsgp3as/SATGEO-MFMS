# 프로젝트 구조 및 개발 가이드
## 목차
1. [algorithm/ 디렉토리 변경](#1-algorithm-디렉토리-변경)
   - [1.1 디렉토리 구조](#11-디렉토리-구조)
   - [1.2 인자 변경](#12-인자-변경)

---
## 1. `algorithm/` 폴더 

### 1.1 디렉토리 구조
```
app/algorithm/
├── 함수코드/
│   ├── code/
│   │   ├── 알고리즘명.py  # 메인 함수
│   │   ├── utils/
│   │   │   ├── utils.py
│   │   │   ├── cfg.py  # 파라미터 정의
│   ├── data/
│   │   ├── input/      # 입력자료 샘플 => 해당 경로 기준으로 입출력 경로 지정
│   │   ├── output/     # 출력자료 샘플 => 해당 경로 기준으로 입출력 경로 지정
│   ├── weights/
│   │   ├── 모델.pt     # 학습 모델
```

- **동적 파라미터 정의**: 알고리즘 파일 내 get_args()를 통해 정의; 함수 실행시 변동될 수 있는 인자 예) 입출력경로, 모델파일 등
- **정적 파라미터 정의**: `algorithm/utils/cfg.py`에 정의; 변동되지 않는 인자 예) 임계값, 고정 파라미터 등 

---

## 2. 코드 구조 

### 2.1 메인 함수
- **주요 함수**: **기존 코드**.
- **process()**: **주요 함수 실행 코드**; 함수 호출시 진입점
  주요 함수를 일련의 순서 하에 실행 
- **__main__**: **커맨드 실행시 진입점**
  ```python
    start_time = time.time()

    args = get_args()

    process(args.input_file, args.output_grd_file, args.output_json_file, args.model_weight_file, args.patch_size, args.channel_list)

    processed_time = time.time() - start_time
    logging.info(f"{processed_time:.2f} seconds")
  ```
  와 같은 형태로 구성

### 2.2 인자 명칭 
- 다음의 요소를 포함하도록 구성
- **input/output/(없음)**: 입출력 데이터 및 폴더를 필요로 하는 경우 명시
- **대상**: 예) ais, grd, slc 등
     - 단, {중요명사}_{수식어구} 의 형태로 구성 ex) facilities_confirmed, vessel_detection 등 
- **file/dir**
- 예) input_grd_file, output_json_file, work_dir, input_dir 등 


