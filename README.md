## 디렉토리 구조

   ```
   absolute-be
   │  .gitignore
   │  main.py
   │  README.md
   │  requirements.txt
   │
   ├─app
   │  ├─api
   │  │  │  dependencies.py
   │  │  │  __init__.py
   │  │  │
   │  │  ├─endpoints
   │  │  │  │  ml_recommend.py
   │  │  │  │  recommend.py
   │  │  │  │  s3.py
   │  │  │  │  stations.py
   │  │  │  │  usage_types.py
   │  │  │  │  __init__.py
   │  │  │  │
   │  │  │  └─__pycache__
   │  │  └─__pycache__
   │  │
   │  ├─comparison
   │  │  │  benchmark.py
   │  │  │  ml_performance_test.py
   │  │  │  performance_test.py
   │  │  │  test_api.py
   │  │  │  __init__.py
   │  │  │
   │  │  ├─algorithms
   │  │  │  │  ahp_topsis.py
   │  │  │  │  base.py
   │  │  │  │  collaborative.py
   │  │  │  │  cosine_similarity.py
   │  │  │  │  euclidean_distance.py
   │  │  │  │  pearson_correlation.py
   │  │  │  │  popularity.py
   │  │  │  │
   │  │  │  └─__pycache__
   │  │  └─__pycache__
   │  │
   │  ├─core
   │  │  │  config.py
   │  │  │  __init__.py
   │  │  │
   │  │  └─__pycache__
   │  │
   │  ├─db
   │  │      __init__.py
   │  │
   │  ├─models
   │  │      gas_station.py
   │  │      recommendation.py
   │  │      usage_type.py
   │  │      __init__.py
   │  │
   │  ├─schemas
   │  │  │  gas_station.py
   │  │  │  recommendation.py
   │  │  │  usage_type.py
   │  │  │  __init__.py
   │  │  │
   │  │  └─__pycache__
   │  │
   │  ├─services
   │  │  │  geo_service.py
   │  │  │  ml_location_recommender.py
   │  │  │  recommend_service.py
   │  │  │  __init__.py
   │  │  │
   │  │  └─__pycache__
   │  │
   │  ├─utils
   │  │  │  data_loader.py
   │  │  │  preprocessing.py
   │  │  │  __init__.py
   │  │  │
   │  │  └─__pycache__
   │  │
   │  └─__pycache__
   │
   ├─data
   │      2024년_도로종류별_교통량_및_XY좌표.csv
   │      gas_station_features.csv
   │      jeonju_gas_station.csv
   │      train.csv
   │      대분류_센터로이드.csv
   │      숙박업소수집계_행정동별.csv
   │      전국1000명당사업체수_행정동별.csv
   │      전국인구수_행정동별.csv
   │      추천결과_행단위.csv
   │      폐주유소좌표변환.csv
   │
   ├─tests
   └─__pycache__
   ```

<br>


## 실행 방법


레포지토리 클론

   ```bash
   git clone https://github.com/culyrh/absolute-BE.git
   cd absolute-be
   ```

필요한 패키지 설치

   ```bash
   pip install -r requirements.txt
   ```

서버 실행

   ```bash
   python main.py
   ```

브라우저에서 엔드포인트 문서 확인해주세요

   http://localhost:8000/docs