# Feature Calculation Dictionary

이 문서는 **risk_engine.py** 에서 사용되는 모든 피처(feature)의 데이터 출처, 적용 기간, 그리고 정확한 계산식(공식)을 설명합니다.

---

## 1. 시계열(시장) 피처
| 피처 | 설명 | 데이터 출처 | 적용 기간 | 계산식 |
|------|------|------------|-----------|--------|
| **Volatility (변동성)** | 일일 수익률의 표준편차에 연간화 계수를 곱한 연율 변동성. | `hist['Close']` (조정 종가) | 최근 **252 거래일**(≈ 1년). 정지 기업인 경우 `cutoff_date` 이전 데이터만 사용. | `pct_changes = closes.pct_change().dropna(); volatility = pct_changes.std() * np.sqrt(252)` |
| **Volume‑Surge (`feat_hype_volume_surge`)** | 기간 내 최고 거래량을 평균 거래량으로 나눈 비율. | `hist['Volume']` | 위와 동일한 252일 기간. | `avg_vol = volumes.mean(); volume_surge = volumes.max() / avg_vol` |
| **Momentum (모멘텀)** | 최근 3개월(≈ 60 거래일) 수익률. 60일 미만이면 전체 구간 사용. | `hist['Close']` | 최대 **60일**(가능하면) 내에서 계산. | `momentum = (closes.iloc[-1] - closes.iloc[-60]) / closes.iloc[-60]` (대체: 전체 구간) |
| **Maximum Draw‑Down (MDD, 최대 낙폭)** | 기간 내 최고점에서 최저점까지의 하락 비율 최댓값. | `hist['Close']` | 252일 전체. | `roll_max = closes.cummax(); drawdown = (closes - roll_max) / roll_max; mdd = abs(drawdown.min())` |
| **High‑Gap (52주 고가 격차)** | 현재 종가와 52주(≈ 1년) 최고가 사이의 비율 차이. | `hist['Close']` | 252일 전체(≈ 52주). | `high_52w = closes.max(); high_gap = (high_52w - closes.iloc[-1]) / high_52w` |

> **참고** – 기업이 `is_suspended` 로 표시되면 `cutoff_date = last_trading_date - 548일`(≈ 18개월) 로 정해진 시점 이전 데이터만 사용합니다. 모든 시계열 피처는 이 제한된 데이터에 대해 계산됩니다.

---

## 2. 재무제표 기반 피처 (연도별)
| 피처 | 설명 | 원본 테이블 | 계산식 |
|------|------|------------|--------|
| **Revenue (매출액)** | 회계연도 전체 매출. | `financials` | `revenue = row_fin.get('Total Revenue', row_fin.get('Operating Revenue', 0))` |
| **Operating Income (영업이익)** | 이자·세금 차감 전 영업이익. | `financials` | `op_income = row_fin.get('Operating Income', row_fin.get('Total Operating Income As Reported', 0))` |
| **Net Income (당기순이익)** | 최종 순이익. | `financials` | `net_income = row_fin.get('Net Income', 0)` |
| **R&D Expense (R&D 투자액)** | 연구·개발 비용. | `financials` | `rd_exp = row_fin.get('Research And Development', 0)` |
| **R&D Ratio (%)** | 매출 대비 R&D 비중. | 파생 | `rd_ratio = (rd_exp / revenue) * 100` |
| **Common Stock (자본금)** | 발행 주식 자본. | `balance_sheet` | `common_stock = row_bs.get('Common Stock', row_bs.get('Capital Stock', 0))` |
| **Total Equity (자본총계)** | 주주지분(순자산). | `balance_sheet` | `total_equity = row_bs.get('Stockholders Equity', row_bs.get('Total Equity Gross Minority Interest', 0))` |
| **Total Assets (자산총계)** | 전체 자산 합계. | `balance_sheet` | `total_assets = row_bs.get('Total Assets', 0)` |
| **Total Liabilities (부채총계)** | 전체 부채 합계. | `balance_sheet` | `total_liab = row_bs.get('Total Liabilities Net Minority Interest', row_bs.get('Total Liabilities', 0))` |
| **Capital Impairment Ratio (자본잠식률)** | (자본금 − 자본총계) ÷ 자본금. 1 > 값은 완전 자본잠식. | 파생 | `impairment_ratio = (common_stock - total_equity) / common_stock` (단, `common_stock > 0`인 경우) |
| **Debt Ratio (%) (부채비율)** | 부채 ÷ 자본총계 × 100. | 파생 | `debt_ratio = (total_liab / total_equity) * 100` (단, `total_equity > 0`인 경우) |
| **Paid‑In Capital Increase (유상증자 조달액)** | 유상증자·주식발행 등으로 조달된 현금. | `cashflow` | `capital_issuance = row_cf.get('Issuance Of Capital Stock', row_cf.get('Common Stock Issuance', row_cf.get('Net Common Stock Issuance', 0)))` |
| **Revenue Growth YoY (%) (연간 매출 성장률)** | 전년 대비 매출 성장률. | 파생 (전년도 행) | `growth = ((curr_rev - prev_rev) / prev_rev) * 100` |
| **Consecutive Low‑Revenue Years (연속 매출 미달 연도)** | 가장 최근 연도부터 매출이 **3 억원 미만**인 연속 연도 수. | 파생 | 최신 연도부터 역순 순회하며 `Revenue < 3_000_000_000` 인 경우 카운트 |
| **Paid‑In Count (3년) (최근 3년 유상증자 횟수)** | 최근 3년 동안 유상증자가 있었던 연도 수. | 파생 | `paid_in_count = sum(Paid_In_Capital_Increase > 0)` (최근 3행) |
| **Paid‑In Amount (3년) (최근 3년 유상증자 총액)** | 최근 3년 간 유상증자 총액. | 파생 | `paid_in_amount = sum(Paid_In_Capital_Increase)` (최근 3행) |
| **Insider Holdings (%) (내부자 지분율)** | 내부자(임원·주요 주주) 보유 비율. | `info['heldPercentInsiders']` (yfinance) | `insider_holdings_pct = heldPercentInsiders * 100` (없을 경우 18.5 %) |
| **Has Dividend (배당 여부)** | 배당을 지급했는지 여부(1=배당, 0=없음). | `info['dividendYield']` 혹은 `info['dividendRate']` | `1 if (dividendYield > 0 or dividendRate > 0) else 0` |
| **Consecutive Operating‑Loss Years (연속 영업손실 연도)** | 영업이익이 연속으로 음수인 년도 수. | 파생 | 최신 연도부터 역순 순회하며 `Operating_Income < 0` 인 경우 카운트 |
| **CB/BW Disclosure Count (전환·신주인수권부사채 공시 횟수)** | DART 공시 중 전환사채(CB) 또는 신주인수권부사채(BW) 언급 건수. | DART 공시 리스트 | `cb_bw_count = sum('전환사채' in report_nm or '신주인수권부사채' in report_nm ...)` |
| **Final Diagnosis (Status) (최종 판별 등급)** | 대시보드에 표시되는 전반적인 위험 라벨 (`거래 정상`, `투자 위험`, `거래 정지`). | 결정 로직 (아래 참고) | **Decision Rules** 섹션 참조 |

---

## 3. 최종 판별 규칙 (Final Status)
| 조건 | 결과 `final_status` |
|------|-------------------|
| `is_suspended` 가 **True** | `거래 정지` |
| `latest_impairment ≥ 1.0` **또는** `latest_equity < 0` | `거래 정지` |
| `latest_impairment ≥ 0.1` **또는** `consecutive_low_rev_years ≥ 3` **또는** `insider_holdings_pct < 15.0` **또는** (`paid_in_count ≥ 2` **그리고** `paid_in_amount > latest_equity * 0.5`) | `투자 위험` |
| 그 외 | `거래 정상` |

---

## 4. 코드에서 사용되는 모든 피처명 (Feature Names)
```
feat_volatility
feat_hype_volume_surge
feat_momentum
feat_mdd
feat_high_gap
Revenue
Operating_Income
Net_Income
R&D_Expense
R&D_Ratio_Pct
Common_Stock
Total_Equity
Total_Assets
Total_Liabilities
Capital_Impairs... (truncated for brevity)
```

---

## 5. 수치 해석 방법
- **비율**(`R&D_Ratio_Pct`, `Debt_Ratio_Pct`, `Capital_Imp...`)은 **백분율**(0 ~ 100 %)로 표시됩니다.
- **카운트**(`paid_in_count`, `cb_bw_count`, `consecutive_low_rev_years` 등)는 정수값입니다.
- **금액**은 **한국 원(KRW)**이며 정수형으로 저장됩니다.
- **시계열 피처**(`volatility`, `momentum`, `mdd`, `high_gap`)는 단위가 없는 소수값이며, `0.12`는 **12 %**에 해당합니다.

---

*본 문서는 2026‑06‑05 기준 `risk_engine.py` 구현에 따라 자동 생성되었습니다.*
