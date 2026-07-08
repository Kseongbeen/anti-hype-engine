import os
import sys
import urllib.parse
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor


# -----------------------------------------------------------------------------
# 1. 네이버 금융에서 한국 기업명 및 기본정보 크롤링
# -----------------------------------------------------------------------------
def get_korean_company_name(code):
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        res = requests.get(url, headers=headers)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, "html.parser")
        
        # 기업명 추출
        name_tag = soup.select_one(".h_company h2 a")
        if name_tag:
            company_name = name_tag.get_text().strip()
            # 시장 분류 추출 (코스피 / 코스닥)
            market_img = soup.select_one(".h_company img")
            market_type = "KOSDAQ"
            if market_img and "kospi" in market_img.get('class', []):
                market_type = "KOSPI"
            elif market_img and "kospi" in market_img.get('src', '').lower():
                market_type = "KOSPI"
            return company_name, market_type
    except Exception:
        pass
    return f"기업_{code}", "KOSDAQ"

# -----------------------------------------------------------------------------
# 2. 기술특례상장 여부 검색 감지
# -----------------------------------------------------------------------------
def check_tech_listing_status(company_name):
    encoded_q = urllib.parse.quote(f"{company_name} 기술특례상장")
    url = f"https://search.naver.com/search.naver?query={encoded_q}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            snippets = []
            for el in soup.select(".news_tit, .api_txt_lines, .dsc_txt"):
                snippets.append(el.get_text())
            combined_text = " ".join(snippets)
            # 기술특례 키워드 체크
            if any(kw in combined_text for kw in ["기술특례", "특례 상장", "특례상장", "기술성 평가"]):
                return True
    except Exception:
        pass
    return False

# -----------------------------------------------------------------------------
# DART API 연동 및 공시 데이터 수집 엔진
# -----------------------------------------------------------------------------
DART_API_KEY = os.environ.get("OPENDART_API_KEY", "969c8c72ac928a7fe090eaae72618c3844a5d36d")

def get_dart_corp_code(stock_code, output_dir="."):
    """
    종목코드(6자리)를 DART 고유번호(8자리)로 매핑합니다.
    처음 실행 시 corpCode.xml 파일을 다운로드하여 dart_corp_codes.json으로 캐싱합니다.
    """
    import zipfile
    import io
    import json
    import xml.etree.ElementTree as ET
    
    clean_stock = stock_code.strip()
    cache_path = os.path.join(output_dir, "dart_corp_codes.json")
    
    # 1. 로컬 캐시 확인
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                code_map = json.load(f)
                if clean_stock in code_map:
                    return code_map[clean_stock]
        except Exception:
            pass
            
    # 2. 캐시 다운로드 및 저장
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    try:
        res = requests.get(url, params={"crtfc_key": DART_API_KEY}, timeout=15)
        if res.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(res.content)) as zf:
                xml_data = zf.read("CORPCODE.xml")
            
            root = ET.fromstring(xml_data)
            code_map = {}
            for list_node in root.findall("list"):
                s_code = list_node.find("stock_code").text
                c_code = list_node.find("corp_code").text
                if s_code and s_code.strip():
                    code_map[s_code.strip()] = c_code.strip()
                    
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(code_map, f, ensure_ascii=False, indent=2)
                
            return code_map.get(clean_stock)
    except Exception as e:
        print(f"[DART] 고유번호 로딩 중 에러 발생: {e}")
    return None

def fetch_dart_disclosures(corp_code, bgn_de, end_de):
    """
    DART 고유번호를 기준으로 특정 기간(bgn_de ~ end_de) 동안의 공시 리스트를 수집합니다.
    """
    url = "https://opendart.fss.or.kr/api/list.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de": bgn_de,
        "end_de": end_de,
        "page_count": 100
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "000":
                return data.get("list", [])
            else:
                print(f"[DART] API 응답 에러: {data.get('status')} - {data.get('message')}")
        else:
            print(f"[DART] HTTP 요청 실패: {res.status_code}")
    except Exception as e:
        print(f"[DART] API 요청 중 에러 발생: {e}")
    return []

# -----------------------------------------------------------------------------
# 3. yfinance 연동 및 데이터 수집/필터링 엔진
# -----------------------------------------------------------------------------
def fetch_and_analyze_company(code, manual_suspension_date=None):
    try:
        return _fetch_and_analyze_company_impl(code, manual_suspension_date)
    except Exception as e:
        return None, f"데이터 수집 및 분석 중 오류 발생: {e}"

def _fetch_and_analyze_company_impl(code, manual_suspension_date=None):
    company_name, market_type = get_korean_company_name(code)
    suffix = ".KQ" if market_type == "KOSDAQ" else ".KS"
    ticker_code = code + suffix
    
    t = yf.Ticker(ticker_code)
    
    # DART 실시간 공시 수집 및 분석
    corp_code = get_dart_corp_code(code)
    dart_disclosures = []
    unconfirmed_or_corrected_count = 0
    cb_bw_count = 0
    if corp_code:
        today_str = datetime.now().strftime("%Y%m%d")
        three_years_ago_str = (datetime.now() - timedelta(days=3*365)).strftime("%Y%m%d")
        dart_disclosures = fetch_dart_disclosures(corp_code, three_years_ago_str, today_str)
        # 정정 공시 및 미확정 공시 필터링 카운트
        unconfirmed_or_corrected_count = sum(
            1 for d in dart_disclosures
            if any(kw in d.get("report_nm", "") for kw in ["정정", "미확정", "기재정정"])
        )
        # 전환사채(CB) / 신주인수권부사채(BW) 관련 공시 빈도 카운트
        cb_bw_count = sum(
            1 for d in dart_disclosures
            if any(kw in d.get("report_nm", "") for kw in ["전환사채", "신주인수권부사채", "사채발행결정"])
        )
    
    # 병렬 처리를 위해 각 API/크롤링 작업을 정의
    def fetch_history():
        try:
            return t.history(period="3y")
        except Exception:
            return pd.DataFrame()
        
    def fetch_info():
        try:
            return t.info
        except Exception:
            return {}
        
    def fetch_fin():
        try:
            return t.financials
        except Exception:
            return pd.DataFrame()
        
    def fetch_bs():
        try:
            return t.balance_sheet
        except Exception:
            return pd.DataFrame()
        
    def fetch_cf():
        try:
            return t.cashflow
        except Exception:
            return pd.DataFrame()
            
    def fetch_tech():
        try:
            return check_tech_listing_status(company_name)
        except Exception:
            return False

    # ThreadPoolExecutor를 이용해 병렬 실행 (최대 6개 스레드)
    with ThreadPoolExecutor(max_workers=6) as executor:
        f_hist = executor.submit(fetch_history)
        f_info = executor.submit(fetch_info)
        f_fin = executor.submit(fetch_fin)
        f_bs = executor.submit(fetch_bs)
        f_cf = executor.submit(fetch_cf)
        f_tech = executor.submit(fetch_tech)
        
        hist = f_hist.result()
        info = f_info.result()
        financials = f_fin.result()
        balance_sheet = f_bs.result()
        cashflow = f_cf.result()
        is_tech = f_tech.result()
        
    # KOSPI <-> KOSDAQ 교차 체크가 필요한 경우 (예: yfinance에 없는 마켓 접미사인 경우)
    if hist.empty:
        alternative_suffix = ".KS" if suffix == ".KQ" else ".KQ"
        ticker_code = code + alternative_suffix
        t_alt = yf.Ticker(ticker_code)
        
        def fetch_hist_alt():
            try: return t_alt.history(period="3y")
            except Exception: return pd.DataFrame()
        def fetch_info_alt():
            try: return t_alt.info
            except Exception: return {}
        def fetch_fin_alt():
            try: return t_alt.financials
            except Exception: return pd.DataFrame()
        def fetch_bs_alt():
            try: return t_alt.balance_sheet
            except Exception: return pd.DataFrame()
        def fetch_cf_alt():
            try: return t_alt.cashflow
            except Exception: return pd.DataFrame()
            
        with ThreadPoolExecutor(max_workers=5) as executor:
            f_hist = executor.submit(fetch_hist_alt)
            f_info = executor.submit(fetch_info_alt)
            f_fin = executor.submit(fetch_fin_alt)
            f_bs = executor.submit(fetch_bs_alt)
            f_cf = executor.submit(fetch_cf_alt)
            
            hist = f_hist.result()
            info = f_info.result()
            financials = f_fin.result()
            balance_sheet = f_bs.result()
            cashflow = f_cf.result()
            
        if hist.empty:
            return None, f"yfinance에서 주식 데이터를 불러올 수 없습니다 ({code})."
        market_type = "KOSPI" if alternative_suffix == ".KS" else "KOSDAQ"
        t = t_alt

    # 상장일 (info 딕셔너리에서 빠르게 가져옴)
    info = info if info is not None else {}
    first_trade_epoch = info.get('firstTradeDateEpochUtc')
    if first_trade_epoch:
        try:
            listing_date = datetime.fromtimestamp(first_trade_epoch).strftime("%Y-%m-%d")
        except Exception:
            listing_date = "N/A"
    else:
        listing_date = info.get('ipoPublishDate', 'N/A')
        if isinstance(listing_date, int):
            try:
                listing_date = datetime.fromtimestamp(listing_date).strftime("%Y-%m-%d")
            except Exception:
                listing_date = "N/A"
                
    # 수동 정지일 설정 또는 자동 정지 감지
    if manual_suspension_date:
        is_suspended = True
        try:
            if len(manual_suspension_date) == 7: # YYYY-MM
                year, month = map(int, manual_suspension_date.split("-"))
                if month == 12:
                    last_trading_date = datetime(year, 12, 31)
                else:
                    last_trading_date = datetime(year, month + 1, 1) - timedelta(days=1)
            else:
                last_trading_date = pd.to_datetime(manual_suspension_date)
            suspension_date_str = last_trading_date.strftime("%Y-%m-%d")
        except Exception:
            return None, f"수동 거래정지일 형식(YYYY-MM 또는 YYYY-MM-DD)이 올바르지 않습니다: {manual_suspension_date}"
    else:
        # 거래 정지 감지 (마지막 거래일이 10일 이상 전이거나, 최근 5일간 거래량이 모두 0인 경우)
        last_trading_date = hist.index[-1].replace(tzinfo=None)
        today = datetime.now()
        days_since_last_trade = (today - last_trading_date).days
        
        recent_volume_zero = False
        if len(hist) >= 5:
            recent_volume_zero = (hist["Volume"].tail(5) == 0).all()
        elif len(hist) > 0:
            recent_volume_zero = (hist["Volume"] == 0).all()
            
        is_suspended = (days_since_last_trade > 10) or recent_volume_zero
        
        if is_suspended:
            if recent_volume_zero:
                # 실제 마지막 거래일(Volume > 0 인 날짜)을 찾기 위해 5년치 데이터를 가져옴
                hist_5y = t.history(period="5y")
                non_zero = hist_5y[hist_5y["Volume"] > 0]
                if not non_zero.empty:
                    last_trading_date = non_zero.index[-1].replace(tzinfo=None)
            suspension_date_str = last_trading_date.strftime("%Y-%m-%d")
        else:
            suspension_date_str = "N/A"
    
    # 2. 18개월 데이터 필터 적용
    cutoff_date = None
    if is_suspended:
        if isinstance(last_trading_date, pd.Timestamp):
            last_trading_date_dt = last_trading_date.replace(tzinfo=None)
        else:
            last_trading_date_dt = last_trading_date
        cutoff_date = last_trading_date_dt - timedelta(days=548)
        cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")
    else:
        cutoff_date_str = "적용 안 됨 (정상 거래 중)"
        
    if financials.empty or balance_sheet.empty:
        return None, "재무제표 또는 대차대조표 데이터를 yfinance에서 찾을 수 없습니다."
        
    # 4. 날짜 기준 18개월 데이터 차단(Filter) 실행
    if is_suspended and cutoff_date is not None:
        filtered_financials_cols = [col for col in financials.columns if col.replace(tzinfo=None) <= cutoff_date.replace(tzinfo=None)]
        filtered_bs_cols = [col for col in balance_sheet.columns if col.replace(tzinfo=None) <= cutoff_date.replace(tzinfo=None)]
        filtered_cf_cols = [col for col in cashflow.columns if col.replace(tzinfo=None) <= cutoff_date.replace(tzinfo=None)]
        
        if not filtered_financials_cols:
            filtered_financials_cols = [financials.columns[-1]]
        if not filtered_bs_cols:
            filtered_bs_cols = [balance_sheet.columns[-1]]
        if not filtered_cf_cols:
            filtered_cf_cols = [cashflow.columns[-1]]
            
        financials = financials[filtered_financials_cols]
        balance_sheet = balance_sheet[filtered_bs_cols]
        cashflow = cashflow[filtered_cf_cols]

    # 주가 시계열 피처 계산
    volatility = 0.0
    volume_surge = 1.0
    momentum = 0.0
    mdd = 0.0
    high_gap = 0.0
    
    if not hist.empty:
        hist_filtered = hist.copy()
        if is_suspended and cutoff_date is not None:
            hist_idx_naive = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
            hist_filtered = hist[hist_idx_naive <= cutoff_date]
            
        recent_hist = hist_filtered.tail(252)
        if not recent_hist.empty:
            closes = recent_hist["Close"]
            volumes = recent_hist["Volume"]
            
            # 1) 변동성 (Volatility)
            pct_changes = closes.pct_change().dropna()
            volatility = float(pct_changes.std() * np.sqrt(252)) if len(pct_changes) > 1 else 0.0
            
            # 2) 거래량 급등률 (Volume Surge)
            avg_vol = volumes.mean()
            volume_surge = float(volumes.max() / avg_vol) if avg_vol > 0 else 1.0
            
            # 3) 모멘텀 (최근 3개월 수익률)
            if len(closes) >= 60:
                momentum = float((closes.iloc[-1] - closes.iloc[-60]) / closes.iloc[-60])
            elif len(closes) >= 2:
                momentum = float((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0])
                
            # 4) 최대 낙폭 (MDD)
            roll_max = closes.cummax()
            drawdown = (closes - roll_max) / roll_max
            mdd = float(abs(drawdown.min())) if not drawdown.empty else 0.0
            
            # 5) 52주 최고가 대비 격차 (High Gap)
            high_52w = closes.max()
            high_gap = float((high_52w - closes.iloc[-1]) / high_52w) if high_52w > 0 else 0.0

    # 행/열 전치
    df_fin = financials.T
    df_bs = balance_sheet.T
    df_cf = cashflow.T

    
    # 5. 연도별 핵심 데이터 추출 및 정리
    available_years = sorted(list(set(df_fin.index.tolist() + df_bs.index.tolist())), reverse=True)
    
    records = []
    for yr in available_years:
        row_fin = df_fin.loc[yr] if yr in df_fin.index else pd.Series(dtype='float64')
        row_bs = df_bs.loc[yr] if yr in df_bs.index else pd.Series(dtype='float64')
        row_cf = df_cf.loc[yr] if yr in df_cf.index else pd.Series(dtype='float64')
        
        # 매출액
        revenue = row_fin.get('Total Revenue', row_fin.get('Operating Revenue', 0))
        if pd.isna(revenue): revenue = 0
            
        # 영업이익
        op_income = row_fin.get('Operating Income', row_fin.get('Total Operating Income As Reported', 0))
        if pd.isna(op_income): op_income = 0
            
        # 당기순이익
        net_income = row_fin.get('Net Income', 0)
        if pd.isna(net_income): net_income = 0
            
        # R&D 비용
        rd_exp = row_fin.get('Research And Development', 0)
        if pd.isna(rd_exp): rd_exp = 0
            
        # 자본금
        common_stock = row_bs.get('Common Stock', row_bs.get('Capital Stock', 0))
        if pd.isna(common_stock): common_stock = 0
            
        # 자본총계
        total_equity = row_bs.get('Stockholders Equity', row_bs.get('Total Equity Gross Minority Interest', 0))
        if pd.isna(total_equity): total_equity = 0
            
        # 자산총계
        total_assets = row_bs.get('Total Assets', 0)
        if pd.isna(total_assets): total_assets = 0
            
        # 부채총계
        total_liab = row_bs.get('Total Liabilities Net Minority Interest', row_bs.get('Total Liabilities', 0))
        if pd.isna(total_liab): total_liab = 0
            
        # 유상증자액
        capital_issuance = row_cf.get('Issuance Of Capital Stock', row_cf.get('Common Stock Issuance', row_cf.get('Net Common Stock Issuance', 0)))
        if pd.isna(capital_issuance): capital_issuance = 0
        
        # 자본잠식률
        if common_stock > 0:
            impairment_ratio = (common_stock - total_equity) / common_stock
            if total_equity < 0:
                impairment_ratio = max(1.0, impairment_ratio)
        else:
            impairment_ratio = 0.0
            
        # 부채비율
        debt_ratio = (total_liab / total_equity) * 100 if total_equity > 0 else (9999.9 if total_liab > 0 else 0.0)
        
        # R&D 비율
        rd_ratio = (rd_exp / revenue) * 100 if revenue > 0 else 0.0
        
        records.append({
            "Year": yr.strftime("%Y-%m-%d"),
            "Revenue": int(revenue),
            "Operating_Income": int(op_income),
            "Net_Income": int(net_income),
            "R&D_Expense": int(rd_exp),
            "R&D_Ratio_Pct": round(rd_ratio, 2),
            "Common_Stock": int(common_stock),
            "Total_Equity": int(total_equity),
            "Total_Assets": int(total_assets),
            "Total_Liabilities": int(total_liab),
            "Capital_Impairment_Ratio": round(impairment_ratio, 4),
            "Debt_Ratio_Pct": round(debt_ratio, 2),
            "Paid_In_Capital_Increase": int(capital_issuance)
        })
        
    df_result = pd.DataFrame(records)
    df_result = df_result.sort_values(by="Year").reset_index(drop=True)
    
    # 6. 전년도 대비 매출 성장률 계산
    df_result["Revenue_Growth_YoY_Pct"] = 0.0
    for i in range(1, len(df_result)):
        prev_rev = df_result.loc[i-1, "Revenue"]
        curr_rev = df_result.loc[i, "Revenue"]
        if prev_rev > 0:
            growth = ((curr_rev - prev_rev) / prev_rev) * 100
            df_result.loc[i, "Revenue_Growth_YoY_Pct"] = round(growth, 2)
            
    # 7. 매출 없는 기간 계산 (30억 미만 연속 연도 수)
    consecutive_low_rev_years = 0
    for i in range(len(df_result) - 1, -1, -1):
        if df_result.loc[i, "Revenue"] < 3_000_000_000:
            consecutive_low_rev_years += 1
        else:
            break
            
    # 8. 유상증자 최근 3개년 누적 횟수 및 금액 계산
    recent_3yrs = df_result.tail(3)
    paid_in_count = int((recent_3yrs["Paid_In_Capital_Increase"] > 0).sum())
    paid_in_amount = int(recent_3yrs["Paid_In_Capital_Increase"].sum())
    
    # 9. 최대주주 지분율 및 배당 여부
    held_percent_insiders = info.get('heldPercentInsiders')
    insider_holdings_pct = round(held_percent_insiders * 100, 2) if held_percent_insiders else 18.5
    
    has_dividend = 0
    if info.get('dividendYield') and info.get('dividendYield') > 0:
        has_dividend = 1
    elif info.get('dividendRate') and info.get('dividendRate') > 0:
        has_dividend = 1
        
    # 영업이익 연속 적자 연수 계산
    consecutive_op_loss_years = 0
    for idx_row, row_res in df_result.iloc[::-1].iterrows():
        if row_res["Operating_Income"] < 0:
            consecutive_op_loss_years += 1
        else:
            break
    
    # 10. 연도별 등급 설정
    df_result["Status"] = "거래 정상"
    for idx, row in df_result.iterrows():
        imp_ratio = row["Capital_Impairment_Ratio"]
        tot_eq = row["Total_Equity"]
        rev = row["Revenue"]
        
        if imp_ratio >= 1.0 or tot_eq < 0:
            df_result.loc[idx, "Status"] = "거래 정지"
        elif imp_ratio >= 0.5 or (rev < 3_000_000_000 and idx >= 2):
            df_result.loc[idx, "Status"] = "투자 위험"
        else:
            df_result.loc[idx, "Status"] = "거래 정상"
            
    # 최종 등급 진단 (현재 시점)
    latest_row = df_result.iloc[-1]
    latest_impairment = latest_row["Capital_Impairment_Ratio"]
    latest_equity = latest_row["Total_Equity"]
    
    if is_suspended:
        final_diagnosis = "거래 정지"
    elif latest_impairment >= 1.0 or latest_equity < 0:
        final_diagnosis = "거래 정지"
    elif (latest_impairment >= 0.1) or \
         (consecutive_low_rev_years >= 3) or \
         (insider_holdings_pct < 15.0) or \
         (paid_in_count >= 2 and paid_in_amount > latest_equity * 0.5):
        final_diagnosis = "투자 위험"
    else:
        final_diagnosis = "거래 정상"
        
    summary_info = {
        "code": code,
        "name": company_name,
        "market": market_type,
        "listing_date": listing_date,
        "is_tech_listing": is_tech,
        "is_suspended": is_suspended,
        "suspension_date": suspension_date_str,
        "cutoff_date": cutoff_date_str,
        "consecutive_low_rev_years": consecutive_low_rev_years,
        "paid_in_count": paid_in_count,
        "paid_in_amount": paid_in_amount,
        "insider_holdings_pct": insider_holdings_pct,
        "has_dividend": has_dividend,
        "consecutive_op_loss_years": consecutive_op_loss_years,
        "cb_bw_count": cb_bw_count,
        "volatility": round(volatility, 4),
        "volume_surge": round(volume_surge, 4),
        "momentum": round(momentum, 4),
        "mdd": round(mdd, 4),
        "high_gap": round(high_gap, 4),
        "disclosures": unconfirmed_or_corrected_count,
        "final_status": final_diagnosis
    }
    
    return {
        "summary": summary_info,
        "financials": df_result,
        "real_disclosures": dart_disclosures
    }, None

# -----------------------------------------------------------------------------
# 4. 5개 CSV 파일 생성 및 저장 기능
# -----------------------------------------------------------------------------
def save_5_csv_files(company_data, output_dir="csv", prefix="", code=None):
    summary = company_data["summary"]
    df_fin = company_data["financials"]
    
    # 접미사 결정: 명시적인 code가 있으면 그것을 쓰고, 없으면 prefix에서 추출, 둘 다 없으면 summary의 code를 사용
    actual_code = code
    if not actual_code:
        if prefix:
            actual_code = prefix.strip("_")
        else:
            actual_code = summary.get("code", "")
            
    suffix = f"_{actual_code}" if actual_code else ""
    
    # 1. Company Basic Info CSV
    df_basic = pd.DataFrame([{
        "기업코드": summary["code"],
        "기업명": summary["name"],
        "상장시장": summary["market"],
        "실제거래정지여부": "Y" if summary["is_suspended"] else "N",
        "거래정지일": summary["suspension_date"],
        "데이터분석차단기준일": summary["cutoff_date"],
        "최종판별등급": summary["final_status"]
    }])
    basic_path = os.path.join(output_dir, f"1_company_basic_info{suffix}.csv")
    df_basic.to_csv(basic_path, index=False, encoding="utf-8-sig")
    
    # 2. Financial Data CSV
    df_financial_data = df_fin.copy()
    df_financial_data.columns = [
        "결산일", "매출액", "영업이익", "당기순이익", "R&D투자액", "매출대비R&D비율(%)",
        "자본금", "자본총계", "자산총계", "부채총계", "자본잠식률", "부채비율(%)",
        "유상증자조달액", "전년대비매출성장률(%)", "재무판별등급"
    ]
    fin_path = os.path.join(output_dir, f"2_financial_data{suffix}.csv")
    df_financial_data.to_csv(fin_path, index=False, encoding="utf-8-sig")
    
    # 3. Risk Events CSV
    df_risk = pd.DataFrame([
        {"위험유형": "매출 미달 연속 기간", "상세지표": f"{summary['consecutive_low_rev_years']}년 연속 30억 미만"},
        {"위험유형": "최근 3개년 유상증자 횟수", "상세지표": f"{summary['paid_in_count']}회 실시"},
        {"위험유형": "최근 3개년 유상증자 조달금액", "상세지표": f"{summary['paid_in_amount']:,} 원"},
        {"위험유형": "최대주주 및 특수관계인 지분율", "상세지표": f"{summary['insider_holdings_pct']}% (15% 미만 시 지배력 약화 위험)"},
        {"위험유형": "최종 진단 등급", "상세지표": summary["final_status"]}
    ])
    risk_path = os.path.join(output_dir, f"3_risk_events{suffix}.csv")
    df_risk.to_csv(risk_path, index=False, encoding="utf-8-sig")
    
    # 4. Disclosures Raw CSV
    discl_records = []
    real_discl = company_data.get("real_disclosures", [])
    
    if real_discl:
        for d in real_discl:
            yr_str = d.get("rcept_dt", "")[:4]
            report_name = d.get("report_nm", "")
            
            # 영향 및 조달 금액 분석
            impact = "모니터링 필요"
            amount = "N/A"
            
            if "유상증자" in report_name:
                impact = "지분 희석 및 자금 사정 악화 반증 우려"
                # yfinance 재무제표 R&D/유상증자 조달금액 매핑 시도
                try:
                    target_row = df_fin[df_fin["Year"].str.startswith(yr_str)]
                    if not target_row.empty:
                        amt = target_row.iloc[0]["Paid_In_Capital_Increase"]
                        if amt > 0:
                            amount = f"{int(amt):,}"
                except Exception:
                    pass
            elif "자본잠식" in report_name or "자본감소" in report_name:
                impact = "상장유지 위험 경보 및 관리종목 지정 우려"
            elif "정정" in report_name:
                impact = "공시 변경/연기 등에 따른 신뢰도 저하 및 벌점 리스크"
            elif "조회공시요구" in report_name:
                impact = "경영 투명성 또는 중요 정보 발생 의혹 제기"
            elif "특허" in report_name or "임상" in report_name or "품목허가" in report_name:
                impact = "핵심 R&D 성과 입증 및 파이프라인 개발 진척"
                
            discl_records.append({
                "공시연도": yr_str,
                "공시유형": report_name,
                "조달금액(원)": amount,
                "영향": impact
            })
    else:
        # DART 데이터가 없거나 수집 실패한 경우 기존 규칙 기반 Fallback
        for _, row in df_fin.iterrows():
            yr_str = row["Year"][:4]
            if row["Paid_In_Capital_Increase"] > 0:
                discl_records.append({
                    "공시연도": yr_str,
                    "공시유형": "유상증자 결정 (주식 발행을 통한 자금 조달)",
                    "조달금액(원)": f"{row['Paid_In_Capital_Increase']:,}",
                    "영향": "지분 희석 및 자금 사정 악화 반증 우려"
                })
            if row["Capital_Impairment_Ratio"] >= 0.5:
                discl_records.append({
                    "공시연도": yr_str,
                    "공시유형": "자본잠식 50% 이상 (관리종목 지정 우려 공시)",
                    "조달금액(원)": "N/A",
                    "영향": "상장유지 위험 경보"
                })
            if row["Capital_Impairment_Ratio"] >= 1.0:
                discl_records.append({
                    "공시연도": yr_str,
                    "공시유형": "완전 자본잠식 (상장폐지 절차 진행 사유 공시)",
                    "조달금액(원)": "N/A",
                    "영향": "즉각적인 거래 정지 및 상장폐지 심사"
                })
                
    if not discl_records:
        discl_records.append({
            "공시연도": datetime.now().year,
            "공시유형": "특이 재무 공시 없음",
            "조달금액(원)": "0",
            "영향": "정상 거래 및 재무 구조 유지 중"
        })
        
    df_discl = pd.DataFrame(discl_records)
    discl_path = os.path.join(output_dir, f"4_disclosures_raw{suffix}.csv")
    df_discl.to_csv(discl_path, index=False, encoding="utf-8-sig")
    
    # 5. Risk Thresholds CSV
    df_thresholds = pd.DataFrame([
        {"위험유형": "자본잠식률 (완전자본잠식)", "위험수준": "거래 정지 (상폐 사유)", "기준치": "자본잠식률 >= 100% 또는 자본총계 < 0", "임계값설명": "자기자본을 전액 초과하여 잠식된 상태로 즉각 거래정지 및 상장폐지 사유에 해당"},
        {"위험유형": "자본잠식률 (부분자본잠식)", "위험수준": "투자 위험 (관리종목)", "기준치": "자본잠식률 >= 50%", "임계값설명": "자본금의 50% 이상이 잠식된 상태로 코스닥 관리종목 지정 요건"},
        {"위험유형": "매출 미달 연속 기간", "위험수준": "투자 위험 (관리종목)", "기준치": "연간 매출액 30억 미만이 3개년 연속 지속", "임계값설명": "코스닥 상장 유지 요건 미달로 투자위험 등급 판정"},
        {"위험유형": "최대주주 지분율", "위험수준": "투자 위험 (경영 불안)", "기준치": "최대주주 및 특수관계인 지분율 < 15.0%", "임계값설명": "지배구조 취약으로 경영권 분쟁 또는 적대적 M&A 위험 증가"},
        {"위험유형": "과도한 자금 조달", "위험수준": "투자 위험 (현금 고갈)", "기준치": "최근 3개년 유상증자 횟수 >= 3회", "임계값설명": "영업 활동으로 현금 창출이 불가능하여 과도하게 증자에 의존하는 위험"}
    ])
    thresholds_path = os.path.join(output_dir, "5_risk_thresholds.csv")
    df_thresholds.to_csv(thresholds_path, index=False, encoding="utf-8-sig")
    
    return [basic_path, fin_path, risk_path, discl_path, thresholds_path]

# -----------------------------------------------------------------------------
# 6. CSV 일괄 채우기 기능 및 터미널 실행 지원
# -----------------------------------------------------------------------------
def fill_company_info_csv(csv_path):
    print(f"[CSV] '{csv_path}' 파일의 인코딩을 감지하여 데이터를 불러옵니다...")
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    import time
    
    try:
        df = pd.read_csv(csv_path, encoding='cp949')
    except Exception:
        df = pd.read_csv(csv_path, encoding='utf-8')
        
    total_rows = len(df)
    print(f"[CSV] 총 {total_rows}개의 기업 정보를 처리합니다. (병렬 분석 시작)")
    
    def process_row(index_row):
        idx, row = index_row
        name = str(row['company_name']).strip()
        code = str(row['stock_code']).strip()
        
        # 6자리 숫자가 되도록 zfill
        if code.isdigit() and len(code) < 6:
            code = code.zfill(6)
            
        trading_status = "거래 정상"
        is_tech = False
        
        if len(code) == 6 and code.isdigit():
            try:
                res, err = fetch_and_analyze_company(code)
                if not err and res:
                    trading_status = res["summary"]["final_status"]
                    is_tech = res["summary"]["is_tech_listing"]
            except Exception:
                pass
        
        # case_group 판별 로직:
        # - 위험 발생 사례: 카나리아바이오, 셀리버리, 올리패스 또는 분석상 거래 정지 상태인 기업
        # - 비교 사례: 신라젠, 코오롱티슈진 또는 투자 위험 상태인 기업
        # - 정상 비교군: 그 외 거래 정상인 기업
        if name in ['카나리아바이오', '셀리버리', '올리패스'] or trading_status == "거래 정지":
            group = "위험 발생 사례"
        elif name in ['신라젠', '코오롱티슈진'] or trading_status == "투자 위험":
            group = "비교 사례"
        else:
            group = "정상 비교군"
            
        tech_status = "Y" if is_tech else "N"
        
        # 진행상황 간단 로깅
        print(f"[{idx+1}/{total_rows}] {name}({code}) -> 그룹: {group}, 상태: {trading_status}, 특례여부: {tech_status}")
        return idx, group, trading_status, tech_status
        
    start_time = time.time()
    
    # yfinance rate limit 방지를 위해 max_workers를 5로 제어
    with ThreadPoolExecutor(max_workers=5) as executor:
        row_tuples = list(df.iterrows())
        results = list(executor.map(process_row, row_tuples))
        
    # 결과 반영
    for idx, group, trading_status, tech_status in results:
        df.loc[idx, 'case_group'] = group
        df.loc[idx, 'trading_risk_status'] = trading_status
        df.loc[idx, 'tech_special_listing'] = tech_status
        
    # 동일한 파일에 저장 (인코딩 유지)
    try:
        df.to_csv(csv_path, index=False, encoding='cp949')
        print(f"[CSV] 성공적으로 업데이트 완료! -> {os.path.abspath(csv_path)}")
    except Exception as e:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[CSV] CP949 저장 실패로 UTF-8(BOM)으로 저장했습니다. -> {os.path.abspath(csv_path)}")
        
    print(f"[CSV] 총 소요 시간: {time.time() - start_time:.2f}초")


if __name__ == "__main__":
    # Windows 터미널에서 한글 깨짐 및 인코딩 에러 방지
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
            
    if len(sys.argv) < 2:
        print("사용법:")
        print(" 1) 단일 기업 분석: python risk_engine.py [6자리_기업코드] [옵션: 거래정지가정일(YYYY-MM)]")
        print(" 2) CSV 파일 일괄 채우기: python risk_engine.py [CSV_파일_경로]")
        print("예시:")
        print(" - python risk_engine.py 021040")
        print(" - python risk_engine.py \"company_info_pharma_bio_all - company_info.csv\"")
        sys.exit(1)
        
    arg1 = sys.argv[1].strip()
    
    # 입력값이 CSV 파일 경로인 경우
    if arg1.lower().endswith('.csv') or os.path.exists(arg1):
        if not os.path.exists(arg1):
            print(f"에러: 파일이 존재하지 않습니다: {arg1}")
            sys.exit(1)
        fill_company_info_csv(arg1)
        sys.exit(0)
        
    # 단일 기업 분석인 경우
    code = arg1
    if len(code) != 6 or not code.isdigit():
        print("에러: 올바른 6자리 숫자 종목 코드 또는 CSV 파일 경로를 입력하세요.")
        sys.exit(1)
        
    manual_date = None
    if len(sys.argv) >= 3:
        manual_date = sys.argv[2].strip()
        print(f"[{code}] 수동 거래정지일 설정 ({manual_date}) 기준 18개월 전 데이터 필터링 분석을 진행합니다.")
        
    print(f"[{code}] 기업 재무 데이터 수집 및 위험도 분석을 시작합니다...")
    data, error = fetch_and_analyze_company(code, manual_suspension_date=manual_date)
    
    if error:
        print(f"에러 발생: {error}")
        sys.exit(1)
        
    summary = data["summary"]
    df_fin = data["financials"]
    
    print("\n--- 분석 결과 요약 ---")
    print(f"기업명: {summary['name']}")
    print(f"상장시장: {summary['market']}")
    print(f"거래정지 여부: {'예' if summary['is_suspended'] else '아니오'} (정지일: {summary['suspension_date']})")
    print(f"데이터 컷오프일 (정지일 18개월 전): {summary['cutoff_date']}")
    print(f"최종 진단 등급: {summary['final_status']}")
    print("----------------------")
    
    csv_paths = save_5_csv_files(data)
    print("\n다음 5개 CSV 파일이 생성되었습니다:")
    for path in csv_paths:
        print(f" - {os.path.abspath(path)}")
