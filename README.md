# 프로젝트 구조 및 개발 가이드
## 목차
1. [algorithm/ 디렉토리 변경](#1-algorithm-디렉토리-변경)
   - [1.1 디렉토리 구조](#11-디렉토리-구조)
   - [1.2 인자 변경](#12-인자-변경)
   - [1.3 DB 서비스 함수 import 및 호출](#13-db-서비스-함수-import-및-호출)
2. [api/routers/ 변경](#2-apirouters-변경)
   - [2.1 인자 변경](#21-인자-변경)
   - [2.2 함수 호출](#22-함수-호출)
   - [2.3 리턴값 지정](#23-리턴값-지정)
3. [models/ 변경](#3-models-변경)
   - [3.1 DB 테이블 정의 변경](#31-db-테이블-정의-변경)
4. [service/ 변경](#4-service-변경)
   - [4.1 DB 처리 함수 정의](#41-db-처리-함수-정의)
5. [환경 파일 (`envs/tb.env`, `envs/local.env`)](#5-환경-파일-envstbenv-envslocalenv)
   - [5.1 인자 전달](#51-인자-전달)

---
## 1. `algorithm/` 디렉토리 변경

### 1.1 디렉토리 구조
```
app/algorithm/
├── 함수코드/
│   ├── 알고리즘명.py  # 메인 함수
│   ├── utils/
│   │   ├── utils.py
│   │   ├── cfg.py  # 파라미터 정의
│   ├── input/      # 입력자료 샘플 => 해당 경로 기준으로 입출력 경로 지정
│   ├── output/     # 출력자료 샘플 => 해당 경로 기준으로 입출력 경로 지정
```

### 1.2 인자 변경
- **입출력 경로 설정**: `config/settings.py`에서 정의 후 알고리즘에서 직접 사용.

#### 코드 예시:
```python
from app.config.settings import settings

def algorithm(db: Session, args: type):
    input_dir = settings.SAMPLE_INPUT_PATH  # 입력 디렉토리
    output_dir = settings.SAMPLE_OUTPUT_PATH  # 출력 디렉토리
    meta_file = settings.SAMPLE_META_FILE  # 메타 파일 경로
```

- **기타 파라미터 정의**: `algorithm/utils/cfg.py`에 정의.
- **DB 테이블 접근**: `db: Session`을 인자로 포함.

### 1.3 DB 서비스 함수 import 및 호출
- DB 관련 작업이 필요한 경우, 관련 서비스를 import 후 호출.

---

## 2. `api/routers/` 변경

### 2.1 인자 변경
- **API 호출 인자**: **Query 처리**.
- **DB 테이블 접근**: 
  ```python
  db: Session = Depends(get_db)
  ```
  를 인자로 포함.

### 2.2 함수 호출
- API가 트리거할 `algorithm/` 하위의 메인 함수를 호출.

### 2.3 리턴값 지정
- 로그에서 확인하거나, wait 처리를 위해 필요한 값을 리턴.

---

## 3. `models/` 변경

### 3.1 DB 테이블 정의 변경
- DB 테이블 정의서를 반영하여 변경.
- 관리자가 작성할 예정이나, 필요시 관리자에게 전달 후 작성.

---

## 4. `service/` 변경

### 4.1 DB 처리 함수 정의
- `models/` 구조를 고려하여 필요한 DB 처리 함수 정의.
- 작성 시:
  - `service/README.md`
  - `sample_service.py`
  에 정의된 사항을 바탕으로 작성.

---

## 5. 환경 파일 (`envs/tb.env`, `envs/local.env`)

### 5.1 인자 전달
- 해당 파일을 직접 변경하지 않고, 본인 알고리즘의 **입출력 폴더 경로**를 관리자에게 전달하여 설정.
