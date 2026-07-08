# Anti-Hype Engine 실행 안내

## 실행 방법

```bash
cd aix_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows에서는 가상환경 활성화 명령만 아래처럼 바꿔 주세요.

```bash
.venv\Scripts\activate
```

## 포함된 설정

- `.streamlit/secrets.toml` 파일이 포함되어 있어 OpenAI API 키 입력 없이 실행할 수 있습니다.
- 앱은 기본적으로 `experiments/rd_feature_expansion_20260611`의 R&D 확장 모델 산출물을 사용합니다.
- 뉴스 감성 분석은 최대 20개 기사 제목을 FinBERT로 분석하며, 정량 리스크 점수에는 반영하지 않는 보조 지표입니다.

## 실행 시 참고

- 첫 실행 시 `transformers`가 FinBERT 모델을 내려받을 수 있어 인터넷 연결이 필요할 수 있습니다.
- Naver 뉴스 검색, yfinance 주가 조회, OpenAI 리포트 생성도 인터넷 연결이 필요합니다.
