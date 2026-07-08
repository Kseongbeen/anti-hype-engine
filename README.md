# 📉 The Anti-Hype Engine

바이오 및 제약 주식의 재무 데이터(DART), 뉴스 감성(FinBERT), 주가 변동성 지표를 종합 분석하여 시장의 과열(Hype) 정도와 실질적인 투자 리스크(거래정지 및 자본잠식 등)를 정량 평가하는 AI 기반 대시보드 플랫폼입니다.

본 프로젝트는 **AI+X 중급 프로젝트** 공동 과제로 개발되었습니다.

---

## 📂 프로젝트 구조

* **[app.py](file:///c:/Users/5174k/Code/202210822/AI+X/AI+X_중급/3/aix_project/app.py)**: Streamlit 기반 웹 어플리케이션 인터페이스 및 시각화 코드
* **[risk_engine.py](file:///c:/Users/5174k/Code/202210822/AI+X/AI+X_중급/3/aix_project/risk_engine.py)**: 시장 지표 및 재무 지표를 토대로 정량적 리스크 점수를 계산하는 코어 연산 엔진
* **[feature_explanation.md](file:///c:/Users/5174k/Code/202210822/AI+X/AI+X_중급/3/aix_project/feature_explanation.md)**: 리스크 판단에 사용되는 15+개 피처(매출 성장률, 자본잠식률, 거래량 급증 등)의 세부 공식 사전
* **[calculate_rf_weights_and_scores.py](file:///c:/Users/5174k/Code/202210822/AI+X/AI+X_중급/3/aix_project/calculate_rf_weights_and_scores.py)**: Random Forest 기법을 적용하여 피처의 중요도 가중치를 학습하는 모듈
* **[stock_sentiment_analysis.ipynb](file:///c:/Users/5174k/Code/202210822/AI+X/AI+X_중급/3/aix_project/stock_sentiment_analysis.ipynb)**: FinBERT 감성 분석 알고리즘 연구 및 모델 검증 노트북
* **[company_pvalue_analysis.ipynb](file:///c:/Users/5174k/Code/202210822/AI+X/AI+X_중급/3/aix_project/company_pvalue_analysis.ipynb)**: 재무 요인들과 거래정지 간 상관관계의 유의성(p-value)을 통계적으로 검정한 분석 노트북

---

## ✨ 핵심 기능

1. **멀티채널 정보 융합 리스크 진단**
   * **yfinance API**: 실시간 주가 분석(연율 변동성, 52주 고가 격차, MDD 등)
   * **DART 전자공시**: 유상증자 이력, 전환사채(CB) 및 신주인수권부사채(BW) 공시 빈도 추적
   * **재무제표 분석**: 3년 연속 매출액 3억 미만 여부, 자본잠식률, 부채비율 계산
2. **FinBERT 기반 뉴스 감성 분석**
   * 최근 Naver 뉴스의 주요 헤드라인 20개를 크롤링한 뒤, HuggingFace의 금융 특화 감성 분석 모델인 **FinBERT**로 긍정/부정 스코어를 추정하여 대중의 Hype(광풍) 감지.
3. **Random Forest 기반 중요도 산출**
   * 다양한 재무/시장 리스크 지표가 실제 거래정지 혹은 투자 경고로 연결되는 기여도(Feature Importance) 분석.
4. **OpenAI GPT-4 기반 AI 리포트 자동 작성**
   * 분석 결과를 취합하여 인간 애널리스트 수준의 종합 투자 경고 리포트를 마크다운 형식으로 자동 퍼블리싱.

---

## 🛠️ 기술 스택

* **Frontend**: `Streamlit`
* **Deep Learning & ML**: `Transformers (FinBERT)`, `Scikit-learn (Random Forest)`, `OpenAI API (GPT-4)`
* **Data Sources**: `yfinance`, `DART Open API`, `Naver News Search API`
* **Data Engineering**: `Pandas`, `Numpy`, `Jupyter Notebook`

---

## 🚀 시작하기 (How to Run)

### 1. 환경 설정 및 가상환경 생성
```bash
python -m venv .venv
# Windows 가상환경 활성화
.venv\Scripts\activate
# 의존성 패키지 설치
pip install -r requirements.txt
```

### 2. Streamlit 대시보드 구동
```bash
streamlit run app.py
```
* **참고**: 첫 구동 시 `transformers` 패키지가 FinBERT 모델 가중치를 자동으로 다운로드하므로 인터넷 연결이 필수적입니다. API 연동 정보는 `.streamlit/secrets.toml`에 안전하게 설정해야 합니다.
