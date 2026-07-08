import os
import sys
import platform
import tempfile
import urllib.request
import urllib.parse
import re
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fpdf import FPDF
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import risk_engine

# =============================================================================
# API 키 기본값 설정 (여기에 키를 입력하면 매번 입력하지 않아도 작동합니다)
# =============================================================================
DEFAULT_OPENAI_API_KEY = ""
DEFAULT_GEMINI_API_KEY = ""
DEFAULT_GROQ_API_KEY = ""

def clean_html(html_str):
    if not html_str:
        return ""
    return "\n".join(line.strip() for line in html_str.splitlines())

def call_llm_api(provider, api_key, model, messages, temperature=0.3):
    import json
    import urllib.request
    import urllib.error
    import ssl
    
    if not api_key:
        return None, "API Key가 설정되지 않았습니다."
        
    if provider == "OpenAI":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
    elif provider == "Groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
    elif provider == "Gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = {
                    "parts": [{"text": content}]
                }
            else:
                gemini_role = "user" if role == "user" else "model"
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })
        
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature
            }
        }
        if system_instruction:
            data["systemInstruction"] = system_instruction
    else:
        return None, f"지원되지 않는 AI 제공사입니다: {provider}"
        
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode("utf-8"), 
        headers=headers, 
        method="POST"
    )
    
    try:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=context) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            if provider in ["OpenAI", "Groq"]:
                return res_data["choices"][0]["message"]["content"], None
            elif provider == "Gemini":
                try:
                    content_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                    return content_text, None
                except (KeyError, IndexError) as e:
                    return None, f"Gemini 응답 형식 오류: {res_data}"
    except urllib.error.HTTPError as e:
        try:
            err_msg = e.read().decode("utf-8")
            err_json = json.loads(err_msg)
            return None, err_json.get("error", {}).get("message", f"HTTP Error {e.code}")
        except Exception:
            return None, f"HTTP Error {e.code}: {e.reason}"
    except Exception as e:
        return None, str(e)

def call_openai_api(api_key, model, messages, temperature=0.3):
    return call_llm_api("OpenAI", api_key, model, messages, temperature)

def parse_combined_csv(file_path):
    if not os.path.exists(file_path):
        return None
    
    content = ""
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except Exception:
            continue
            
    if not content:
        return None
        
    sections = {}
    current_section = None
    lines = []
    
    for line in content.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue
        if line_strip.startswith("===") and line_strip.endswith("==="):
            if current_section:
                sections[current_section] = lines
            current_section = line_strip
            lines = []
        else:
            lines.append(line_strip)
    if current_section:
        sections[current_section] = lines
        
    return sections

import io
def parse_section_to_dataframe(lines):
    if not lines:
        return pd.DataFrame()
    csv_str = "\n".join(lines)
    try:
        return pd.read_csv(io.StringIO(csv_str))
    except Exception:
        return pd.DataFrame()

def load_company_data_from_combined_csv(clean_code):
    combined_path = f"csv_combined/{clean_code}_combined.csv"
    if not os.path.exists(combined_path):
        return None, "File not found"
        
    sections = parse_combined_csv(combined_path)
    if not sections:
        return None, "Parse failed"
        
    df_basic = pd.DataFrame()
    df_fin = pd.DataFrame()
    df_risk = pd.DataFrame()
    df_discl = pd.DataFrame()
    
    for key, lines in sections.items():
        if "1." in key or "기본 정보" in key:
            df_basic = parse_section_to_dataframe(lines)
        elif "2." in key or "재무 데이터" in key:
            df_fin = parse_section_to_dataframe(lines)
        elif "3." in key or "위험" in key:
            df_risk = parse_section_to_dataframe(lines)
        elif "4." in key or "공시" in key:
            df_discl = parse_section_to_dataframe(lines)
            
    if df_basic.empty or df_fin.empty:
        return None, "Required sections are empty"
        
    company_name = df_basic.iloc[0]["기업명"] if not df_basic.empty else f"기업_{clean_code}"
    market = df_basic.iloc[0]["상장시장"] if not df_basic.empty else "KOSDAQ"
    
    # Parse risk events
    consecutive_low_rev_years = 0
    paid_in_count = 0
    paid_in_amount = 0
    insider_holdings_pct = 18.5
    
    if not df_risk.empty:
        for _, r_row in df_risk.iterrows():
            v_type = str(r_row.get("위험유형", ""))
            v_desc = str(r_row.get("상세지표", ""))
            if "매출 미달" in v_type or "매출액 미달" in v_type:
                import re
                match_num = re.search(r"\d+", v_desc)
                if match_num:
                    consecutive_low_rev_years = int(match_num.group(0))
            elif "유상증자 횟수" in v_type:
                import re
                match_num = re.search(r"\d+", v_desc)
                if match_num:
                    paid_in_count = int(match_num.group(0))
            elif "유상증자 조달금액" in v_type or "유상증자 발행" in v_type:
                clean_num = ''.join(filter(str.isdigit, v_desc))
                if clean_num:
                    paid_in_amount = float(clean_num)
            elif "지분율" in v_type:
                import re
                match_num = re.search(r"[\d\.]+", v_desc)
                if match_num:
                    insider_holdings_pct = float(match_num.group(0))
                    
    # Map financials
    col_mapping = {
        "결산일": "Year",
        "매출액": "Revenue",
        "영업이익": "Operating_Income",
        "당기순이익": "Net_Income",
        "R&D투자액": "R&D_Expense",
        "매출대비R&D비율(%)": "R&D_Ratio_Pct",
        "자본금": "Common_Stock",
        "자본총계": "Total_Equity",
        "자산총계": "Total_Assets",
        "부채총계": "Total_Liabilities",
        "자본잠식률": "Capital_Impairment_Ratio",
        "부채비율(%)": "Debt_Ratio_Pct",
        "유상증자조달액": "Paid_In_Capital_Increase",
        "전년대비매출성장률(%)": "Revenue_Growth_YoY_Pct",
        "재무판별등급": "Status"
    }
    
    mapped_cols = {}
    for col in df_fin.columns:
        if col in col_mapping:
            mapped_cols[col] = col_mapping[col]
            
    df_fin_mapped = df_fin.rename(columns=mapped_cols)
    
    for eng_col in col_mapping.values():
        if eng_col not in df_fin_mapped.columns:
            df_fin_mapped[eng_col] = 0.0 if eng_col != "Year" and eng_col != "Status" else ("2025-12-31" if eng_col == "Year" else "거래 정상")
            
    op_losses = df_fin_mapped["Operating_Income"].values
    consecutive_op_loss_years = 0
    for val in reversed(op_losses):
        if val < 0:
            consecutive_op_loss_years += 1
        else:
            break
            
    cb_bw_count = 0
    if not df_discl.empty:
        col_type = '공시유형' if '공시유형' in df_discl.columns else df_discl.columns[1]
        cb_bw_count = sum(
            1 for _, d_row in df_discl.iterrows()
            if any(kw in str(d_row.get(col_type, "")).upper() for kw in ["CB", "BW", "전환사채", "신주인수권부사채"])
        )
        
    disclosures_count = 0
    if not df_discl.empty:
        col_type = '공시유형' if '공시유형' in df_discl.columns else df_discl.columns[1]
        disclosures_count = sum(
            1 for _, d_row in df_discl.iterrows()
            if any(kw in str(d_row.get(col_type, "")).lower() for kw in ["정정", "미확정", "기재정정"])
        )
        
    is_tech_listing = True
    is_suspended = False
    suspension_date = "N/A"
    cutoff_date = "N/A"
    
    if not df_basic.empty:
        susp_val = str(df_basic.iloc[0].get("실제거래정지여부", "N")).strip().upper()
        is_suspended = (susp_val == "Y")
        suspension_date = str(df_basic.iloc[0].get("거래정지일", "N/A"))
        cutoff_date = str(df_basic.iloc[0].get("데이터분석차단기준일", "N/A"))
        
    if not df_database.empty and clean_code in df_database['clean_code'].values:
        db_row = df_database[df_database['clean_code'] == clean_code].iloc[0]
        is_tech_listing = bool(db_row.get('is_tech_listing', True))
        
    summary = {
        "code": clean_code,
        "name": company_name,
        "market": market,
        "listing_date": "N/A",
        "is_tech_listing": is_tech_listing,
        "is_suspended": is_suspended,
        "suspension_date": suspension_date,
        "cutoff_date": cutoff_date,
        "consecutive_low_rev_years": consecutive_low_rev_years,
        "paid_in_count": paid_in_count,
        "paid_in_amount": paid_in_amount,
        "insider_holdings_pct": insider_holdings_pct,
        "final_status": df_basic.iloc[0]["최종판별등급"] if not df_basic.empty else "거래 정상",
        "consecutive_op_loss_years": consecutive_op_loss_years,
        "cb_bw_count": cb_bw_count,
        "disclosures": disclosures_count
    }
    
    discl_records = []
    if not df_discl.empty:
        col_year = '공시연도' if '공시연도' in df_discl.columns else df_discl.columns[0]
        col_type = '공시유형' if '공시유형' in df_discl.columns else df_discl.columns[1]
        col_amt = '조달금액(원)' if '조달금액(원)' in df_discl.columns else df_discl.columns[2]
        col_imp = '영향' if '영향' in df_discl.columns else df_discl.columns[3]
        for _, d_row in df_discl.iterrows():
            discl_records.append({
                "Year": d_row.get(col_year, ""),
                "Report_Name": d_row.get(col_type, ""),
                "Amount": d_row.get(col_amt, ""),
                "Impact": d_row.get(col_imp, "")
            })
            
    company_data = {
        "summary": summary,
        "financials": df_fin_mapped,
        "real_disclosures": discl_records
    }
    
    return company_data, None

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 디자인 정의 (Premium Design)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Anti-Hype Engine 대시보드",
    page_icon="",
    layout="wide"
)

#  화면 테마 선택 (사이드바 렌더링)
is_dark = False

# Custom CSS for Premium Design (Dynamic Theme)
if is_dark:
    st.markdown("""
    <style>
        .stApp {
            background-color: #0f172a !important;
            color: #f8fafc !important;
        }
        [data-testid="stSidebar"] {
            background-color: #1e293b !important;
        }
        .main {
            background-color: #0f172a !important;
        }
        .report-card {
            background: #1e293b !important;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
            margin-bottom: 20px;
            border: 1px solid #334155 !important;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #38bdf8 !important;
        }
        .metric-label {
            font-size: 14px;
            color: #94a3b8 !important;
        }
        .warning-box {
            background-color: #4c0519 !important;
            border-left: 5px solid #f43f5e !important;
            padding: 16px;
            border-radius: 6px;
            color: #fecdd3 !important;
            margin-bottom: 20px;
        }
        .undervalued-box {
            background-color: #052e16 !important;
            border-left: 5px solid #22c55e !important;
            padding: 16px;
            border-radius: 6px;
            color: #dcfce7 !important;
            margin-bottom: 20px;
        }
        .fair-box {
            background-color: #172554 !important;
            border-left: 5px solid #3b82f6 !important;
            padding: 16px;
            border-radius: 6px;
            color: #dbeafe !important;
            margin-bottom: 20px;
        }
        h1, h2, h3, h4, h5, h6, p, li, span, label {
            color: #f8fafc !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        .main {
            background-color: #f7f9fc;
        }
        .report-card {
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border: 1px solid #eef2f6;
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            color: #1e3d59;
        }
        .metric-label {
            font-size: 14px;
            color: #64748b;
        }
        .warning-box {
            background-color: #fff1f2;
            border-left: 5px solid #f43f5e;
            padding: 16px;
            border-radius: 6px;
            color: #9f1239;
            margin-bottom: 20px;
        }
        .undervalued-box {
            background-color: #f0fdf4;
            border-left: 5px solid #22c55e;
            padding: 16px;
            border-radius: 6px;
            color: #166534;
            margin-bottom: 20px;
        }
        .fair-box {
            background-color: #eff6ff;
            border-left: 5px solid #3b82f6;
            padding: 16px;
            border-radius: 6px;
            color: #1e40af;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

def render_statistical_significance_dashboard():
    from scipy import stats
    st.header(" 전수 조사 기반 폭락 vs 정상 기업 통계 분석 (P-value 검증)")
    st.markdown("전체 225개 기업 중 성공적으로 재무 데이터를 수집한 **219개 기업**을 대상으로 통계 검정을 실시하여 폭락 유무에 따라 실질적인 지표 차이가 발생하는지 분석합니다.")
    st.markdown("---")
    
    # 1. Load data
    csv_all_path = 'csv/company_info_pharma_bio_all_filled.csv'
    csv_flat_path = 'csv/combined_financial_data_flat.csv'
    
    if not os.path.exists(csv_all_path) or not os.path.exists(csv_flat_path):
        st.error("분석에 필요한 CSV 파일이 존재하지 않습니다. 먼저 데이터 수집 및 전처리를 진행해 주세요.")
        return
        
    try:
        df_all = pd.read_csv(csv_all_path, encoding='cp949')
    except Exception:
        df_all = pd.read_csv(csv_all_path, encoding='utf-8')
        
    df_flat = pd.read_csv(csv_flat_path, encoding='utf-8-sig')
    
    df_all['clean_code'] = df_all['stock_code'].astype(str).str.zfill(6)
    df_flat['clean_code'] = df_flat['기업코드'].astype(str).str.replace('=', '').str.replace('"', '').str.zfill(6)
    
    df_merged = pd.merge(
        df_flat, 
        df_all[['clean_code', 'drop_50_or_more_since_2025', 'drop_pct_since_2025', 'years_after_listing']], 
        on='clean_code', 
        how='inner'
    )
    
    g1 = df_merged[df_merged['drop_50_or_more_since_2025'] == 1]
    g0 = df_merged[df_merged['drop_50_or_more_since_2025'] == 0]
    
    # Display overall metrics
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.markdown(f"""
        <div class="report-card">
            <div class="metric-label">총 분석 기업 수</div>
            <div class="metric-value">{len(df_merged)} 개</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div class="report-card">
            <div class="metric-label">폭락 기업군 (Drop &ge; 50%)</div>
            <div class="metric-value" style="color: #ef4444;">{len(g1)} 개 ({len(g1)/len(df_merged):.1%})</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat3:
        st.markdown(f"""
        <div class="report-card">
            <div class="metric-label">정상 기업군 (Drop &lt; 50%)</div>
            <div class="metric-value" style="color: #22c55e;">{len(g0)} 개 ({len(g0)/len(df_merged):.1%})</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader(" 1. 상장 시장 구분(KOSPI vs KOSDAQ) 독립성 검정")
    
    contingency_table = pd.crosstab(df_all['market'], df_all['drop_50_or_more_since_2025'])
    
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.write("**시장구분 및 폭락여부 분할표 (Contingency Table)**")
        formatted_table = contingency_table.copy()
        formatted_table.columns = ['정상 기업', '폭락 기업']
        formatted_table.index.name = '시장'
        st.dataframe(formatted_table, use_container_width=True)
        
        try:
            chi2, chi2_pval, dof, expected = stats.chi2_contingency(contingency_table)
            st.markdown(f"""
            - **카이제곱 통계량**: `{chi2:.4f}` (자유도: {dof})
            - **카이제곱 검정 P-value**: `{chi2_pval:.6f}`
            """)
            if chi2_pval < 0.05:
                st.success(" **해석**: 시장 구분과 폭락 상태 간에는 **통계적으로 유의미한 상관관계**가 존재합니다. (KOSDAQ 기업이 KOSPI 기업보다 폭락할 확률이 유의미하게 높음)")
            else:
                st.info(" **해석**: 시장 구분과 폭락 상태 간에는 통계적으로 유의미한 상관관계가 발견되지 않았습니다. (p &ge; 0.05)")
        except Exception as e:
            st.error(f"카이제곱 독립성 검정 중 오류 발생: {e}")
            
    with col_t2:
        if os.path.exists("significant_market_distribution.png"):
            st.image("significant_market_distribution.png", caption="시장별 폭락/정상 기업 비율 비교", use_container_width=True)
        else:
            st.info("상관관계 차트 파일이 없습니다.")
            
    st.subheader(" 2. 수치형 재무 및 인프라 지표 차이 검정")
    st.markdown("각 지표에 대해 두 집단의 평균 및 중앙값 차이를 비교하고, **Welch's t-test** 및 **Mann-Whitney U test**를 통해 P-value를 산출했습니다. 표는 Mann-Whitney U 검정 P-value 기준 오름차순으로 정렬되었습니다.")
    
    exclude_cols = ['기업코드', '기업명', '상장시장', '실제거래정지여부', '거래정지일', '데이터차단기준일(18m)', 'clean_code', 'drop_50_or_more_since_2025', 'drop_pct_since_2025']
    numeric_features = []
    for col in df_merged.columns:
        if col not in exclude_cols and not col.endswith('_결산일') and not col.endswith('_재무판별등급'):
            if pd.api.types.is_numeric_dtype(df_merged[col]):
                numeric_features.append(col)
                
    results = []
    for feat in numeric_features:
        val1 = g1[feat].dropna()
        val0 = g0[feat].dropna()
        if len(val1) < 2 or len(val0) < 2:
            continue
        mean1 = val1.mean()
        mean0 = val0.mean()
        median1 = val1.median()
        median0 = val0.median()
        
        t_stat, t_pval = stats.ttest_ind(val1, val0, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(val1, val0, alternative='two-sided')
        
        results.append({
            'Feature': feat,
            'Mean_Crash': mean1,
            'Mean_NonCrash': mean0,
            'Median_Crash': median1,
            'Median_NonCrash': median0,
            'MW_PValue': u_pval if not np.isnan(u_pval) else 1.0,
            'T_PValue': t_pval if not np.isnan(t_pval) else 1.0
        })
        
    df_res = pd.DataFrame(results).sort_values(by='MW_PValue').reset_index(drop=True)
    
    display_df = df_res.copy()
    display_df.columns = ['지표명', '폭락군 평균', '정상군 평균', '폭락군 중앙값', '정상군 중앙값', 'MW_PValue (U검정)', 'T_PValue (T검정)']
    
    def highlight_pvals(val):
        color = 'white'
        if isinstance(val, float):
            if val < 0.01:
                color = '#e0f2fe'
            elif val < 0.05:
                color = '#f0fdf4'
        return f'background-color: {color}'
        
    styled_df = display_df.style.map(highlight_pvals, subset=['MW_PValue (U검정)', 'T_PValue (T검정)'])
    
    format_fn = lambda x: "" if pd.isna(x) else (f"{x:,.2f}" if abs(x) < 10000 else f"{x:,.0f}")
    format_dict = {
        '폭락군 평균': format_fn,
        '정상군 평균': format_fn,
        '폭락군 중앙값': format_fn,
        '정상군 중앙값': format_fn,
        'MW_PValue (U검정)': '{:.2e}',
        'T_PValue (T검정)': '{:.2e}'
    }
    
    styled_df = styled_df.format(format_dict)
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    st.markdown("""
    <div style="display: flex; gap: 20px; font-size: 13px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 15px; height: 15px; background-color: #e0f2fe; border: 1px solid #93c5fd; border-radius: 3px;"></div>
            <span>극도로 유의미함 (p < 0.01)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 5px;">
            <div style="width: 15px; height: 15px; background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 3px;"></div>
            <span>통계적으로 유의미함 (p < 0.05)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader(" 3. 개별 지표 상세 분포 탐색기")
    st.markdown("위 표의 지표 중 하나를 선택하면 상세 통계 비교 및 분포 시각화 차트를 실시간 렌더링합니다.")
    
    selected_feat = st.selectbox("분석할 지표 선택", df_res['Feature'].tolist())
    feat_stats = df_res[df_res['Feature'] == selected_feat].iloc[0]
    
    col_feat1, col_feat2 = st.columns([1, 1])
    
    with col_feat1:
        st.markdown(f"#### **{selected_feat} 통계 비교**")
        val1 = g1[selected_feat].dropna()
        val0 = g0[selected_feat].dropna()
        
        def format_val(val):
            if '액(원)' in selected_feat or '비용' in selected_feat or '금(원)' in selected_feat or '계(원)' in selected_feat or '조달액(원)' in selected_feat:
                return f"{val / 100000000:,.1f} 억 원"
            elif '비율' in selected_feat or '률' in selected_feat or '%' in selected_feat:
                return f"{val:.2f} %"
            elif '연수' in selected_feat or 'listing' in selected_feat:
                return f"{val:.1f} 년"
            else:
                return f"{val:,.1f}"
                
        st.markdown(f"""
        - **폭락 기업군 (Crash)**
          - 평균: `{format_val(feat_stats['Mean_Crash'])}`
          - 중앙값: `{format_val(feat_stats['Median_Crash'])}`
          - 표준편차: `{format_val(val1.std())}`
        - **정상 기업군 (Non-Crash)**
          - 평균: `{format_val(feat_stats['Mean_NonCrash'])}`
          - 중앙값: `{format_val(feat_stats['Median_NonCrash'])}`
          - 표준편차: `{format_val(val0.std())}`
        """)
        
        st.markdown("#### **가설 검정 결과**")
        mwp = feat_stats['MW_PValue']
        tp = feat_stats['T_PValue']
        st.markdown(f"""
        * **Mann-Whitney U 검정 P-value**: `{mwp:.2e}`
        * **Welch's t-검정 P-value**: `{tp:.2e}`
        """)
        
        if mwp < 0.01:
            st.success(f" **검정 결과**: 폭락 기업군과 정상 기업군 간의 {selected_feat} 분포 차이는 **통계적으로 극도로 유의미**합니다. (p < 0.01)")
        elif mwp < 0.05:
            st.success(f" **검정 결과**: 폭락 기업군과 정상 기업군 간의 {selected_feat} 분포 차이는 **통계적으로 유의미**합니다. (p < 0.05)")
        else:
            st.warning(f" **검정 결과**: 폭락 기업군과 정상 기업군 간의 {selected_feat} 분포 차이는 **통계적으로 유의미한 차이가 없습니다.** (p >= 0.05)")
            
    with col_feat2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=df_merged,
            x='drop_50_or_more_since_2025',
            y=selected_feat,
            palette=['#3b82f6', '#ef4444'],
            ax=ax,
            width=0.4
        )
        sns.stripplot(
            data=df_merged,
            x='drop_50_or_more_since_2025',
            y=selected_feat,
            color='black',
            alpha=0.3,
            size=4,
            jitter=0.15,
            ax=ax
        )
        
        ax.set_xticklabels(['정상 기업', '폭락 기업'])
        ax.set_xlabel('기업 분류')
        
        ylabel = selected_feat
        if '(원)' in selected_feat:
            ylabel = selected_feat.replace('(원)', ' (억원)')
            ticks = ax.get_yticks()
            ax.set_yticklabels([f"{val/100000000:,.0f}" for val in ticks])
            
        ax.set_ylabel(ylabel, weight='bold')
        ax.set_title(f"{selected_feat} 분포 비교 박스플롯", fontsize=11, weight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def render_global_market_matrix():
    st.header("️ 225개 제약/바이오 기업 글로벌 마켓 매트릭스 (Hype vs Substance)")
    st.markdown("정성평가 및 8대 변수를 기반으로 도출된 전체 **225개 제약/바이오 기업**의 실체 역량(Substance)과 시장 과열도(Hype)를 교차 플로팅하여 입체적으로 조망합니다.")
    st.markdown("---")
    
    csv_final_path = '마지막/점수화 해본 결과.csv'
    import re
    year_match = re.search(r"(\d{4})년", analysis_year)
    year_str = year_match.group(1) if year_match else "2025"
    if year_str == "2024":
        scores_file = "company_substance_scores_rf.csv"
    else:
        scores_file = f"company_substance_scores_rf_{year_str}.csv"
    csv_scores_path = f"csv/{scores_file}"
    
    if not os.path.exists(csv_final_path):
        st.error("분석에 필요한 최종 점수화 결과 CSV 파일이 존재하지 않습니다.")
        return
        
    try:
        df_final = pd.read_csv(csv_final_path, encoding='utf-8')
    except Exception:
        df_final = pd.read_csv(csv_final_path, encoding='cp949')
        
    df_final['stock_code'] = df_final['stock_code'].astype(str).str.zfill(6)
    
    # ML 스코어 데이터 병합
    has_ml = False
    if os.path.exists(csv_scores_path):
        try:
            df_scores = pd.read_csv(csv_scores_path)
            df_scores['stock_code'] = df_scores['stock_code'].astype(str).str.zfill(6)
            # 중복 컬럼 방지를 위해 필요한 스코어만 머지
            df_final = pd.merge(df_final, df_scores[['stock_code', 'substance_score_rf', 'hype_score_rf', 'rnd_score_rf', 'hype_index_rf']], on='stock_code', how='left')
            df_final['ML_재무리스크점수'] = df_final['substance_score_rf'] * 100
            df_final['ML_HYPE리스크점수'] = df_final['hype_score_rf'] * 100
            df_final['ML_R&D리스크점수'] = df_final['rnd_score_rf'] * 100
            df_final['ML_종합붕괴위험도'] = df_final['hype_index_rf'] * 100
            
            # ML 사분면 분류 계산
            def get_ml_quadrant(row):
                x_val = row.get('ML_재무리스크점수')
                y_val = row.get('ML_HYPE리스크점수')
                z_val = row.get('ML_R&D리스크점수')
                if pd.isna(x_val) or pd.isna(y_val) or pd.isna(z_val):
                    return "분류 불가"
                
                if x_val < 40 and y_val < 40 and z_val < 40:
                    return "양호 (저위험 기업군)"
                
                max_val = max(x_val, y_val, z_val)
                if max_val >= 60:
                    if max_val == y_val:
                        return "HYPE 과열 위험군"
                    elif max_val == x_val:
                        return "재무 부실 위험군"
                    else:
                        return "R&D/기술 위험군"
                return "보통 (중립 기업군)"
            
            df_final['ML_사분면분류'] = df_final.apply(get_ml_quadrant, axis=1)
            has_ml = True
        except Exception as e:
            st.warning(f"ML 스코어 데이터를 병합하는 데 실패했습니다: {e}")

    # 1. 시각화 모드 선택 토글
    st.subheader(" 시각화 분석 기준 선택")
    st.markdown("수기 정성평가는 특정 지표가 나쁘면 반대 지표도 무조건 나쁠 것이라는 인간의 편향(Bias) 때문에 극단적인 음의 상관관계(일직선 대각선)로 몰리는 경향이 있습니다. 머신러닝 모델은 각 지표를 객관적으로 다차원 결합하여 현실적인 4분면 기업 분포를 보여줍니다.")
    
    modes = ["1) 팀원 정성평가 점수 매트릭스 (수기 평정 바이어스 존재)"]
    if has_ml:
        modes.append("2) 머신러닝 SOTA 리스크 매트릭스 (3D 리스크 버블 차트)")
        
    plot_mode = st.radio(
        "분석 기준 선택",
        options=modes,
        horizontal=True
    )
    
    if "팀원 정성평가" in plot_mode:
        x_col = "실체_역량총점"
        y_col = "시장_과열지수총점"
        x_label = "실체 역량 총점 (Substance)"
        y_label = "시장 과열 지수 총점 (Hype)"
        quadrant_col = "종합사분면분류"
    else:
        x_col = "ML_HYPE리스크점수"
        y_col = "ML_재무리스크점수"
        x_label = "ML 시장 과열 리스크 (Hype Risk)"
        y_label = "ML 기업 부실 리스크 (Financial Risk)"
        quadrant_col = "ML_사분면분류"
        df_final = df_final.dropna(subset=[x_col, y_col])
        
    # 2. 사이드바 필터 구성 (글로벌 매트릭스용)
    st.sidebar.subheader("️ 글로벌 매트릭스 필터")
    
    # 시장 필터
    markets = df_final['market'].dropna().unique().tolist()
    selected_markets = st.sidebar.multiselect("시장 구분 선택", options=markets, default=markets)
    
    # 케이스 분류 필터
    case_groups = df_final['case_group'].dropna().unique().tolist()
    selected_cases = st.sidebar.multiselect("분석 케이스 선택", options=case_groups, default=case_groups)
    
    # 사분면 분류 필터
    quadrants = df_final[quadrant_col].dropna().unique().tolist()
    selected_quadrants = st.sidebar.multiselect("종합 사분면 선택", options=quadrants, default=quadrants)
    
    # 필터 적용
    df_filtered = df_final[
        (df_final['market'].isin(selected_markets)) &
        (df_final['case_group'].isin(selected_cases)) &
        (df_final[quadrant_col].isin(selected_quadrants))
    ].copy()
    
    if df_filtered.empty:
        st.warning("선택한 필터 조건에 부합하는 기업이 존재하지 않습니다.")
        return
        
    # 3. 상단 핵심 요약 지표 카드
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="report-card">
            <div class="metric-label">현재 보기 기업 수</div>
            <div class="metric-value">{len(df_filtered)} 개</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        bubble_count = len(df_filtered[df_filtered[quadrant_col].str.contains('투기성 거품', na=False)])
        st.markdown(f"""
        <div class="report-card">
            <div class="metric-label">투기성 거품군 (Low Cap, High Hype)</div>
            <div class="metric-value" style="color: #ef4444;">{bubble_count} 개 ({bubble_count/len(df_filtered):.1%})</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        undervalued_count = len(df_filtered[df_filtered[quadrant_col].str.contains('가치 저평가', na=False)])
        st.markdown(f"""
        <div class="report-card">
            <div class="metric-label">가치 저평가군 (High Cap, Low Hype)</div>
            <div class="metric-value" style="color: #22c55e;">{undervalued_count} 개 ({undervalued_count/len(df_filtered):.1%})</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        fair_count = len(df_filtered[df_filtered[quadrant_col].str.contains('우량 과열|적정 가치', na=False)])
        st.markdown(f"""
        <div class="report-card">
            <div class="metric-label">적정 가치 고성장군 / 우량 과열군</div>
            <div class="metric-value" style="color: #3b82f6;">{fair_count} 개 ({fair_count/len(df_filtered):.1%})</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 4. Plotly Scatter Plot 생성
    color_map = {
        "위험 발생 사례": "#ef4444",   # Red
        "비교 사례": "#f97316",       # Orange
        "정상 비교군": "#22c55e"       # Green
    }
    
    size_col = None
    if "팀원 정성평가" not in plot_mode:
        df_filtered["ML_R&D리스크크기"] = df_filtered["ML_R&D리스크점수"].fillna(30.0).clip(lower=15.0)
        size_col = "ML_R&D리스크크기"
        
    hover_dict = {
        "stock_code": True,
        "market": True,
        quadrant_col: True,
        x_col: ":.1f",
        y_col: ":.1f",
        "case_group": False
    }
    if size_col:
        hover_dict["ML_R&D리스크점수"] = ":.1f"
        
    fig = px.scatter(
        df_filtered,
        x=x_col,
        y=y_col,
        color="case_group",
        size=size_col,
        size_max=18,
        color_discrete_map=color_map,
        hover_name="company_name",
        hover_data=hover_dict,
        title=f"전체 제약/바이오 기업 가치 교차 분석 매트릭스 ({'정성평가 기준' if '팀원 정성평가' in plot_mode else '머신러닝 정량 스코어 기준 (버블 크기: R&D 리스크)'})",
        labels={
            x_col: x_label,
            y_col: y_label
        }
    )
    
    # 마커 크기 및 테두리 설정으로 프리미엄 룩 완성
    fig.update_traces(
        marker=dict(size=11, line=dict(width=1.2, color='black'), opacity=0.85),
        selector=dict(mode='markers')
    )
    
    # 4분면 파스텔톤 배경색 입히기
    fig.add_shape(type="rect", x0=0, y0=50, x1=50, y1=100, fillcolor="#FFCCCC", opacity=0.15, layer="below", line_width=0) # 좌상: 투기성 거품
    fig.add_shape(type="rect", x0=50, y0=0, x1=100, y1=50, fillcolor="#CCFFCC", opacity=0.15, layer="below", line_width=0) # 우하: 저평가 우량
    fig.add_shape(type="rect", x0=50, y0=50, x1=100, y1=100, fillcolor="#CCE5FF", opacity=0.12, layer="below", line_width=0) # 우상: 적정/고성장
    fig.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50, fillcolor="#E0E0E0", opacity=0.15, layer="below", line_width=0) # 좌하: 비우량/소외
    
    # 4분면 십자 기준선 추가
    fig.add_hline(y=50, line_dash="dash", line_color="gray", line_width=1.5)
    fig.add_vline(x=50, line_dash="dash", line_color="gray", line_width=1.5)
    
    text_color_bubble = "#f87171" if is_dark else "#b91c1c"
    text_color_fair = "#60a5fa" if is_dark else "#1d4ed8"
    text_color_undervalued = "#4ade80" if is_dark else "#15803d"
    text_color_neglected = "#94a3b8" if is_dark else "#4b5563"

    # 사분면 라벨 텍스트 레이아웃 추가 (bold 속성 오류 방지 위해 html b 태그 적용)
    fig.add_annotation(x=15, y=95, text="<b> 투기성 거품군</b><br><b>(Low Cap, High Hype)</b>", showarrow=False, font=dict(size=12, color=text_color_bubble))
    fig.add_annotation(x=85, y=95, text="<b> 우량 과열군 / 적정 가치군</b><br><b>(High Cap, High Hype)</b>", showarrow=False, font=dict(size=12, color=text_color_fair))
    fig.add_annotation(x=85, y=5, text="<b> 가치 저평가군</b><br><b>(High Cap, Low Hype)</b>", showarrow=False, font=dict(size=12, color=text_color_undervalued))
    fig.add_annotation(x=15, y=5, text="<b> 정체/소외군</b><br><b>(Low Cap, Low Hype)</b>", showarrow=False, font=dict(size=12, color=text_color_neglected))
    
    # 레이아웃 프리미엄화
    if is_dark:
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(range=[-3, 103], showgrid=True, gridcolor='#334155'),
            yaxis=dict(range=[-3, 103], showgrid=True, gridcolor='#334155'),
            plot_bgcolor='#1e293b',
            paper_bgcolor='#0f172a',
            legend=dict(
                title=dict(text="분석 집단"),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=80, b=40, l=40, r=40),
            height=700
        )
    else:
        fig.update_layout(
            xaxis=dict(range=[-3, 103], showgrid=True, gridcolor='#eef2f6'),
            yaxis=dict(range=[-3, 103], showgrid=True, gridcolor='#eef2f6'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                title=dict(text="분석 집단"),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=80, b=40, l=40, r=40),
            height=700
        )
    
    # 렌더링
    st.plotly_chart(fig, use_container_width=True)
    
    # 수기 평가의 한계점과 ML 활용의 당위성을 설명하는 디자인 박스 추가
    if "팀원 정성평가" in plot_mode:
        st.markdown(clean_html("""
        <div class="warning-box" style="margin-top: 15px;">
            <h4>️ 정성평가(수기 평정) 그래프의 한계점 (Halo Effect / 이분법적 Bias)</h4>
            <ul>
                <li><b>극단적 대각선(반비례) 분포의 원인:</b> 수기 평가 시 평가자들은 "실체가 강한 기업은 시장에서 조용할 것(Hype가 낮음)", 반대로 "시장 거품이 낀 기업은 실체가 없을 것(Substance가 낮음)"이라는 이분법적 선입견에 갇히게 됩니다.</li>
                <li><b>현실과의 불일치:</b> 셀트리온이나 유한양행 등 실제 시장을 선도하는 상위 바이오 기업은 <b>실체(Substance)와 Hype가 동시에 높은 우상향(우상단) 영역</b>에 위치해야 하고, 반대로 시장에서 완전히 소외된 좀비 기업들은 <b>좌하단 영역</b>에 위치해야 하지만, 수기 평가 결과에는 이 두 영역이 텅 비어 있습니다.</li>
                <li><b>해결책:</b> 상단의 토글 스위치에서 <b>'머신러닝(8대 피처) 정량 스코어'</b>를 선택해 보십시오. 다차원 재무 공시 지표를 객관적으로 결합해 자연스러운 4분면 기업 분포를 실시간으로 확인하실 수 있습니다.</li>
            </ul>
        </div>
        """), unsafe_allow_html=True)
    else:
        st.markdown(clean_html("""
        <div class="undervalued-box" style="margin-top: 15px;">
            <h4> 머신러닝(8대 피처) 객관적 가중치 반영의 효과</h4>
            <ul>
                <li><b>자연스러운 4분면 분포:</b> 2025년 이후 주가 50% 폭락 여부라는 실제 정답 데이터(Ground Truth)를 Random Forest 모델로 학습하여 산출한 8대 핵심 변수 가중치를 반영했습니다.</li>
                <li><b>이분법적 선입견 극복:</b> 실체가 높으면서도 시장 관심(Hype)을 지속적으로 받는 <b>우상단(우량 과열군/적정 가치군)</b> 기업들과, 실체와 과열도가 모두 극히 낮은 <b>좌하단(정체/소외군)</b> 기업들이 객관적으로 포착되어 현실적인 분포를 보입니다.</li>
                <li><b>핵심 지표 가중치 스키마:</b>
                    <ul>
                        <li><b>실체(Substance) 가중치:</b> 자산대비영업이익(ebitat: 30%) + 매출액(log_sale: 25%) + 상장연수(listing_years: 24%) + 자기자본비율(seqat: 21%)</li>
                        <li><b>과열도(Hype) 가중치:</b> 시장과열도(log_psr: 53%) + R&D비율(rd: 21%) + 유상증자빈도(financing: 18%) + 자본잠식률(impairment: 8%)</li>
                    </ul>
                </li>
            </ul>
        </div>
        """), unsafe_allow_html=True)

    # 5. 상세 테이블 제공
    st.subheader(" 매트릭스 매칭 상세 데이터 시트")
    display_cols = ['company_name', 'stock_code', 'market', 'case_group', x_col, y_col, quadrant_col]
    df_table = df_filtered[display_cols].copy()
    df_table.columns = ['기업명', '종목코드', '상장시장', '케이스분류', '실체 역량 점수', '시장 과열 지수', '종합 사분면 분류']
    st.dataframe(df_table, use_container_width=True, height=350)


st.title(" Anti-Hype Engine: 기업 가치 기만 및 저평가 탐지 시스템")
st.markdown("DART 공시상의 **실체적 R&D 역량(Substance)**과 실시간 뉴스 감성 및 주가를 반영한 **시장 과열도(Hype)**를 교차 검증하여 분석합니다.")

# -----------------------------------------------------------------------------
# 0. 데이터 분석 기준 연도 선택 (사이드바 최상단)
# -----------------------------------------------------------------------------
analysis_year = st.sidebar.radio(
    " 데이터 분석 기준 연도 선택", 
    [
        "2022년 기준 (2022 결산 데이터)",
        "2023년 기준 (2023 결산 데이터)",
        "2024년 기준 (2024 결산 데이터)",
        "2025년 기준 (2025 결산 데이터)"
    ],
    index=3
)
st.sidebar.markdown("---")

# AI API Configuration
st.sidebar.subheader(" AI 모델 및 API 설정")
ai_provider = st.sidebar.selectbox(
    "AI 제공사 선택",
    ["OpenAI", "Gemini", "Groq"],
    index=["OpenAI", "Gemini", "Groq"].index(st.session_state.get("ai_provider", "OpenAI"))
)
st.session_state["ai_provider"] = ai_provider

# Load environment keys or default constants
default_openai_key = os.environ.get("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY)
default_gemini_key = os.environ.get("GEMINI_API_KEY", DEFAULT_GEMINI_API_KEY)
default_groq_key = os.environ.get("GROQ_API_KEY", DEFAULT_GROQ_API_KEY)

# 3개의 API Key 입력 창 모두 노출
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key 입력", 
    type="password", 
    value=st.session_state.get("openai_api_key", default_openai_key)
)
if openai_api_key:
    st.session_state["openai_api_key"] = openai_api_key

gemini_api_key = st.sidebar.text_input(
    "Gemini API Key 입력", 
    type="password", 
    value=st.session_state.get("gemini_api_key", default_gemini_key)
)
if gemini_api_key:
    st.session_state["gemini_api_key"] = gemini_api_key

groq_api_key = st.sidebar.text_input(
    "Groq API Key 입력", 
    type="password", 
    value=st.session_state.get("groq_api_key", default_groq_key)
)
if groq_api_key:
    st.session_state["groq_api_key"] = groq_api_key

# 선택한 제공사에 따른 모델 선택 및 active 설정
if ai_provider == "OpenAI":
    openai_model = st.sidebar.selectbox(
        "AI 모델 선택",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        index=0
    )
    active_api_key = openai_api_key
    active_model = openai_model
elif ai_provider == "Gemini":
    gemini_model = st.sidebar.selectbox(
        "AI 모델 선택",
        ["gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0
    )
    active_api_key = gemini_api_key
    active_model = gemini_model
else: # Groq
    groq_model = st.sidebar.selectbox(
        "AI 모델 선택",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0
    )
    active_api_key = groq_api_key
    active_model = groq_model

# -----------------------------------------------------------------------------
# 메뉴 선택 (개별 기업 분석 고정, 글로벌 매트릭스 메뉴 삭제)
# -----------------------------------------------------------------------------
menu = "개별 기업 위험도 분석"

# -----------------------------------------------------------------------------
# 2. 모델 로딩 및 프리셋 데이터베이스 정의 (캐싱 적용)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_sentiment_classifier():
    model_name = "snunlp/KR-FinBert-SC"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier = load_sentiment_classifier()

# Load feature explanation for RAG
def get_feature_doc():
    try:
        with open(os.path.join(os.path.dirname(__file__), "feature_explanation.md"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"피처 설명 파일을 읽는 중 오류: {e}")
        return ""

feature_doc = get_feature_doc()

# 데이터베이스 캐싱 로드 함수 (225개 전체 제약/바이오 기업)
@st.cache_data
def load_pharma_bio_database(analysis_year):
    import re
    year_match = re.search(r"(\d{4})년", analysis_year)
    year_str = year_match.group(1) if year_match else "2025"
    if year_str == "2024":
        scores_file = "company_substance_scores_rf.csv"
    else:
        scores_file = f"company_substance_scores_rf_{year_str}.csv"
    scores_path = f"csv/{scores_file}"
    flat_path = "csv/combined_financial_data_flat.csv"
    all_path = "csv/company_info_pharma_bio_all_filled.csv"
    stock_path = "csv/historical_stock_data.csv"
    
    if os.path.exists(scores_path) and os.path.exists(flat_path) and os.path.exists(all_path):
        try:
            # Read df_scores with encoding detection
            df_scores = None
            for enc in ["utf-8-sig", "cp949", "utf-8"]:
                try:
                    df_scores = pd.read_csv(scores_path, encoding=enc)
                    if 'company_name' in df_scores.columns:
                        break
                except Exception:
                    continue
            if df_scores is None:
                df_scores = pd.read_csv(scores_path)
                
            # Read df_flat with encoding detection
            df_flat = None
            for enc in ["utf-8-sig", "cp949", "utf-8"]:
                try:
                    df_flat = pd.read_csv(flat_path, encoding=enc)
                    if '기업코드' in df_flat.columns or '회사명' in df_flat.columns:
                        break
                except Exception:
                    continue
            if df_flat is None:
                df_flat = pd.read_csv(flat_path)
                
            # Read df_all
            df_all = None
            for enc in ["cp949", "utf-8-sig", "utf-8"]:
                try:
                    df_all = pd.read_csv(all_path, encoding=enc)
                    if 'stock_code' in df_all.columns:
                        break
                except Exception:
                    continue
            if df_all is None:
                df_all = pd.read_csv(all_path)
                
            # Read df_stock (historical_stock_data.csv)
            df_stock = None
            if os.path.exists(stock_path):
                for enc in ["utf-8-sig", "cp949", "utf-8"]:
                    try:
                        df_stock = pd.read_csv(stock_path, encoding=enc)
                        if 'sharesOutstanding' in df_stock.columns:
                            break
                    except Exception:
                        continue
                if df_stock is None:
                    df_stock = pd.read_csv(stock_path)
                    
            df_flat['clean_code'] = df_flat['기업코드'].astype(str).str.replace('=', '').str.replace('"', '').str.zfill(6)
            df_all['clean_code'] = df_all['stock_code'].astype(str).str.zfill(6)
            df_scores['clean_code'] = df_scores['stock_code'].astype(str).str.zfill(6)
            
            # 병합
            df_m = pd.merge(df_flat, df_all[['clean_code', 'market', 'case_group', 'years_after_listing']], on='clean_code', how='inner')
            df_m = pd.merge(df_m, df_scores, on='clean_code', how='inner', suffixes=('', '_dup'))
            
            # stock data 병합 (2021_Close, 2022_Close, 2023_Close, 2024_Close, 2025_Close, sharesOutstanding 등)
            if df_stock is not None:
                df_stock['clean_code'] = df_stock['clean_code'].astype(str).str.zfill(6)
                stock_cols = ['clean_code', 'sharesOutstanding', '2021_Close', '2022_Close', '2023_Close', '2024_Close', '2025_Close']
                # filter stock_cols to only exist in df_stock
                stock_cols = [c for c in stock_cols if c in df_stock.columns]
                df_m = pd.merge(df_m, df_stock[stock_cols], on='clean_code', how='left', suffixes=('', '_stockdup'))
            
            # 필요 중복 컬럼 정리
            df_m = df_m.loc[:, ~df_m.columns.str.endswith('_dup')]
            df_m = df_m.loc[:, ~df_m.columns.str.endswith('_stockdup')]
            return df_m
        except Exception as e:
            st.warning(f"데이터베이스 로드 중 오류 발생: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_resource
def train_optimal_ensemble():
    feat_path = "csv/processed_features_2024.csv"
    if not os.path.exists(feat_path):
        return None, None, None
        
    df = pd.read_csv(feat_path)
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    
    X = df[feature_cols]
    X.index = df['clean_code'].values
    y = df['target'].fillna(0)
    
    from sklearn.ensemble import RandomForestClassifier
    
    # 최적 SOTA 파라미터 조합
    ensemble = RandomForestClassifier(
        n_estimators=66,
        max_depth=6,
        min_samples_split=7,
        min_samples_leaf=11,
        max_features='sqrt',
        bootstrap=True,
        max_samples=0.6120142352927195,
        ccp_alpha=0.01400839082255636,
        criterion='entropy',
        random_state=42
    )
    ensemble.fit(X, y)
    
    medians = df.median(numeric_only=True)
    return ensemble, medians, X

@st.cache_data
def load_features_for_year(analysis_year):
    import re
    year_match = re.search(r"(\d{4})년", analysis_year)
    year_str = year_match.group(1) if year_match else "2025"
    feat_file = f"processed_features_{year_str}.csv"
    path = f"csv/{feat_file}"
    if os.path.exists(path):
        df = pd.read_csv(path)
        feature_cols = [c for c in df.columns if c.startswith('feat_')]
        X = df[feature_cols]
        X.index = df['clean_code'].astype(str).str.zfill(6).values
        return X
    return None

df_database = load_pharma_bio_database(analysis_year)
ensemble_model, df_medians, X_train_df = train_optimal_ensemble()
df_features_year = load_features_for_year(analysis_year)

# Matplotlib 한글 폰트 설정 Helper
def setup_plt_font():
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif platform.system() == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

setup_plt_font()

# -----------------------------------------------------------------------------
# 3. 실시간 뉴스 크롤링 함수 (네이버 뉴스 연동)
# -----------------------------------------------------------------------------
def crawl_naver_news(company_name):
    try:
        encoded_query = urllib.parse.quote(company_name)
        url = f"https://search.naver.com/search.naver?where=news&query={encoded_query}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'html.parser')
        
        titles = []
        for a in soup.find_all('a'):
            href = a.get('href', '')
            text = a.get_text().strip()
            if not href.startswith('http'):
                continue
            if len(text) >= 12 and len(text) <= 80:
                exclude_keywords = [
                    '네이버', '로그인', '회원가입', '고객센터', '이용약관', 
                    '개인정보처리방침', '도움말', '바로가기', '쇼핑', '블로그', 
                    '카페', '사전', '지식in', '웹툰', '학술', '뉴스', '증권',
                    '부동산', '지도', '도서', 'keep에 저장', '공유하기'
                ]
                if any(kw in text.lower() for kw in exclude_keywords):
                    continue
                exclude_domains = ['help.naver.com', 'policy.naver.com', 'nid.naver.com']
                if any(d in href for d in exclude_domains):
                    continue
                text = ' '.join(text.split())
                if text not in titles:
                    titles.append(text)
        return titles[:20]
    except Exception as e:
        st.warning(f"실시간 뉴스 검색 실패: {e}")
        return []

# -----------------------------------------------------------------------------
# 4. Naver Search 기반 실시간 R&D 데이터 스크래핑 및 정보 파싱
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Naver Search 실시간 R&D 데이터 스크래핑 중...")
def scrape_rd_data(company_name):
    # 기본값 설정
    rd_staff = 100
    phd_count = 20
    disclosures = 1
    
    queries = [
        f"{company_name} 연구인력 총원",
        f"{company_name} 박사 연구원 수",
        f"{company_name} 미확정 공시 횟수"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    all_text_blocks = []
    try:
        import requests
        for q in queries:
            url = f"https://search.naver.com/search.naver?query={urllib.parse.quote(q)}"
            try:
                res = requests.get(url, headers=headers, timeout=5)
                if res.status_code == 200:
                    soup = BeautifulSoup(res.text, "html.parser")
                    for el in soup(["script", "style", "noscript", "meta"]):
                        el.decompose()
                    text = soup.get_text(separator=" ")
                    all_text_blocks.append(text)
            except Exception:
                continue
                
        combined_text = " ".join(all_text_blocks)
        combined_text = re.sub(r'\s+', ' ', combined_text) # normalize spaces
        
        # 2. Extract Candidates
        rd_staff_candidates = []
        phd_candidates = []
        disclosure_candidates = []
        
        # Check for "명" (people)
        for m in re.finditer(r'(\d+)\s*명', combined_text):
            val = int(m.group(1))
            # Get context
            start_idx = max(0, m.start() - 60)
            end_idx = min(len(combined_text), m.end() + 60)
            context = combined_text[start_idx:end_idx].lower()
            
            # Check for R&D Staff
            if val > 5:
                if any(kw in context for kw in ["연구", "r&d", "개발", "전담", "인력"]):
                    rd_staff_candidates.append(val)
                    
            # Check for PhDs
            if any(kw in context for kw in ["박사", "ph.d", "phd"]):
                phd_candidates.append(val)
                
        # Check for "회" or "건" (disclosures)
        for m in re.finditer(r'(\d+)\s*(?:회|건)', combined_text):
            val = int(m.group(1))
            start_idx = max(0, m.start() - 60)
            end_idx = min(len(combined_text), m.end() + 60)
            context = combined_text[start_idx:end_idx].lower()
            
            if any(kw in context for kw in ["공시", "미확정", "정정", "연기", "반복"]):
                disclosure_candidates.append(val)
                
        # 3. Select Best Values
        if rd_staff_candidates:
            rd_staff = max(set(rd_staff_candidates), key=rd_staff_candidates.count)
        if phd_candidates:
            phd_count = max(set(phd_candidates), key=phd_candidates.count)
        if disclosure_candidates:
            disclosures = max(set(disclosure_candidates), key=disclosure_candidates.count)
            
        is_scraped = bool(rd_staff_candidates or phd_candidates or disclosure_candidates)
        return rd_staff, phd_count, disclosures, is_scraped
    except Exception:
        return rd_staff, phd_count, disclosures, False

def get_company_info(code_str, analysis_year):
    # 숫자만 추출
    clean_code = ''.join(filter(str.isdigit, code_str))
    if not clean_code or len(clean_code) != 6:
        return None, False

    import re
    year_match = re.search(r"(\d{4})년", analysis_year)
    year_str = year_match.group(1) if year_match else "2025"
    y_pref = f"{year_str}_"

    # A. csv_combined에 있는 경우 우선적으로 로드 (오프라인/거래정지 종목 지원)
    combined_path = f"csv_combined/{clean_code}_combined.csv"
    if os.path.exists(combined_path):
        sections = parse_combined_csv(combined_path)
        if sections:
            df_basic = pd.DataFrame()
            df_fin = pd.DataFrame()
            df_risk = pd.DataFrame()
            df_discl = pd.DataFrame()
            
            for key, lines in sections.items():
                if "1." in key or "기본 정보" in key:
                    df_basic = parse_section_to_dataframe(lines)
                elif "2." in key or "재무 데이터" in key:
                    df_fin = parse_section_to_dataframe(lines)
                elif "3." in key or "위험" in key:
                    df_risk = parse_section_to_dataframe(lines)
                elif "4." in key or "공시" in key:
                    df_discl = parse_section_to_dataframe(lines)
            
            company_name = df_basic.iloc[0]["기업명"] if not df_basic.empty else f"기업_{clean_code}"
            market = df_basic.iloc[0]["상장시장"] if not df_basic.empty else "KOSDAQ"
            suffix = ".KS" if str(market).strip().upper() == "KOSPI" else ".KQ"
            
            fin_row = pd.Series()
            data_unavailable = False
            if not df_fin.empty:
                col_date = '결산일' if '결산일' in df_fin.columns else df_fin.columns[0]
                match_rows = df_fin[df_fin[col_date].astype(str).str.startswith(year_str)]
                if not match_rows.empty:
                    fin_row = match_rows.iloc[0]
                else:
                    data_unavailable = True
                    fin_row = pd.Series(dtype='float64')
            else:
                data_unavailable = True
            
            revenue_val = float(fin_row.get("매출액", 10e9))
            rd_ratio = float(fin_row.get("매출대비R&D비율(%)", 8.0))
            impairment_val = float(fin_row.get("자본잠식률", 0.0))
            
            # EBITAT, SEQAT 계산
            assets = float(fin_row.get("자산총계", 1.0))
            if pd.isna(assets) or assets <= 0:
                assets = 1.0
            op_inc = float(fin_row.get("영업이익", 0.0))
            equity = float(fin_row.get("자본총계", 0.0))
            
            ebitat_val = op_inc / assets
            seqat_val = equity / assets
            
            # 위험 요인 파싱
            financing_count = 0
            if not df_risk.empty:
                for _, r_row in df_risk.iterrows():
                    v_type = str(r_row.get("위험유형", ""))
                    v_desc = str(r_row.get("상세지표", ""))
                    if "유상증자 횟수" in v_type:
                        import re
                        match_num = re.search(r"\d+", v_desc)
                        if match_num:
                            financing_count = int(match_num.group(0))
            
            # 데이터베이스 정보 백업 쿼리
            db_row = None
            if not df_database.empty and clean_code in df_database['clean_code'].values:
                db_row = df_database[df_database['clean_code'] == clean_code].iloc[0]
                
            if db_row is not None:
                listing_years = float(db_row.get('years_after_listing', 5.0)) + (float(year_str) - 2024.0)
                try:
                    close_val = float(db_row.get(f'{y_pref}Close', 0.0))
                    shares = float(db_row.get('sharesOutstanding', 0.0))
                    mcap = close_val * shares
                    psr_val = mcap / max(1.0, revenue_val)
                except Exception:
                    psr_val = 1.0
            else:
                listing_years = 5.0
                psr_val = 2.0
                
            # 공시 건수 세기 (해당 선택 연도만 필터링)
            disclosures_count = 0
            if not df_discl.empty:
                col_year = df_discl.columns[0]
                col_type = '공시유형' if '공시유형' in df_discl.columns else df_discl.columns[1]
                try:
                    target_year_int = int(year_str)
                except ValueError:
                    target_year_int = 2025
                
                for _, d_row in df_discl.iterrows():
                    try:
                        row_year_int = int(float(str(d_row.get(col_year, "")).strip()))
                    except ValueError:
                        continue
                    if row_year_int == target_year_int:
                        if any(kw in str(d_row.get(col_type, "")).lower() for kw in ["정정", "미확정", "기재정정"]):
                            disclosures_count += 1
            
            return {
                "name": company_name,
                "suffix": suffix,
                "revenue": revenue_val,
                "rd_ratio": rd_ratio,
                "financing": financing_count,
                "impairment": impairment_val,
                "ebitat": ebitat_val,
                "seqat": seqat_val,
                "psr": psr_val,
                "listing_years": listing_years,
                "disclosures": disclosures_count,
                "rd_staff": 100,
                "phd_count": 20,
                "is_scraped": False,
                "data_unavailable": data_unavailable
            }, True

    # 1. df_database에 등록된 프리셋 기업인 경우 (225개 중 해당 종목 검색)
    if not df_database.empty and clean_code in df_database['clean_code'].values:
        row = df_database[df_database['clean_code'] == clean_code].iloc[0]
        suffix = ".KS" if str(row['market']).strip() == "KOSPI" else ".KQ"
        
        # 3개년 유증 횟수 계산
        try:
            curr_y = int(year_str)
        except ValueError:
            curr_y = 2025
        
        paid_in = 0
        for y_offset in [0, 1, 2]:
            y_check = curr_y - y_offset
            if y_check >= 2021:
                col_name = f'{y_check}_유상증자조달액(원)'
                if col_name in row:
                    if float(row.get(col_name, 0)) > 0:
                        paid_in += 1
        
        # PSR 계산
        try:
            close_val = float(row.get(f'{y_pref}Close', 0))
            shares = float(row.get('sharesOutstanding', 0))
            mcap = close_val * shares
            rev_val = float(row.get(f'{y_pref}매출액(원)', 0))
            psr = mcap / max(1.0, rev_val)
        except Exception:
            psr = 1.0
            
        # DB 프리셋에서도 해당 연도 데이터 존재 여부를 확인
        _rev_check = float(row.get(f'{y_pref}매출액(원)', 0.0))
        _col_exists = f'{y_pref}매출액(원)' in row.index if hasattr(row, 'index') else f'{y_pref}매출액(원)' in row
        _db_unavailable = (not _col_exists) or (pd.isna(_rev_check))
        
        return {
            "name": row['company_name'],
            "suffix": suffix,
            "revenue": float(row.get(f'{y_pref}매출액(원)', 0.0)),
            "rd_ratio": float(row.get(f'{y_pref}매출대비R&D비율(%)', 0.0)),
            "financing": paid_in,
            "impairment": float(row.get(f'{y_pref}자본잠식률(%)', 0.0)),
            "ebitat": float(row.get(f'{y_pref}EBITAT', 0.0)),
            "seqat": float(row.get(f'{y_pref}SEQAT', 0.5)),
            "psr": psr,
            "listing_years": float(row.get('years_after_listing', 5.0)) + (float(year_str) - 2024.0),
            "disclosures": int(row.get('disclosures', 0)),
            "rd_staff": 100,
            "phd_count": 20,
            "is_scraped": False,
            "data_unavailable": _db_unavailable
        }, True

    # 2. 미등록 종목인 경우 Naver / yfinance 매칭
    name, market_type = risk_engine.get_korean_company_name(clean_code)
    suffix = ".KQ" if market_type == "KOSDAQ" else ".KS"
    
    if name == f"기업_{clean_code}":
        for test_suffix in [".KS", ".KQ"]:
            ticker_code = clean_code + test_suffix
            try:
                t = yf.Ticker(ticker_code)
                if not t.history(period="1d").empty:
                    name_yf = t.info.get("shortName", t.info.get("longName", f"기업_{clean_code}"))
                    name = name_yf.split(" ")[0].split(".")[0]
                    suffix = test_suffix
                    break
            except Exception:
                continue

    # 실시간 크롤링 시도
    rd_staff, phd_count, disclosures, is_scraped = scrape_rd_data(name)
    
    try:
        corp_code = risk_engine.get_dart_corp_code(clean_code)
        if corp_code:
            from datetime import datetime, timedelta
            today_str = datetime.now().strftime("%Y%m%d")
            three_years_ago_str = (datetime.now() - timedelta(days=3*365)).strftime("%Y%m%d")
            dart_list = risk_engine.fetch_dart_disclosures(corp_code, three_years_ago_str, today_str)
            disclosures = sum(
                1 for d in dart_list
                if any(kw in d.get("report_nm", "") for kw in ["정정", "미확정", "기재정정"])
                and str(d.get("rcept_dt", "")).startswith(year_str)
            )
            is_scraped = True
    except Exception:
        pass
    
    return {
        "name": name,
        "suffix": suffix,
        "revenue": 10e9, # 기본 100억
        "rd_ratio": 8.0,
        "financing": 0,
        "impairment": 0.0,
        "ebitat": 0.0,
        "seqat": 0.5,
        "psr": 2.0,
        "listing_years": 5.0,
        "disclosures": disclosures,
        "rd_staff": rd_staff,
        "phd_count": phd_count,
        "is_scraped": is_scraped,
        "data_unavailable": False
    }, False

def refine_sentiment(headline, label, score):
    label = label.lower()
    if label == 'label_0':
        label = 'negative'
    elif label == 'label_1':
        label = 'neutral'
    elif label == 'label_2':
        label = 'positive'
        
    hl_lower = headline.lower()
    neutral_kws = ['설명회', '단신', '안내', '일정', '게시', '세미나', '포럼', '주총', 'ir', '개최', '간담회', '공시']
    pos_kws = ['상승', '호재', '급등', '돌파', '체결', '승인', '흑자', '강세', '성공', '기술이전', '수출', '최고가']
    neg_kws = ['폭락', '급락', '적자', '반토막', '소송', '논란', '위기', '악재', '의혹', '해지', '실패', '하락', '약세']
    
    has_neutral = any(kw in hl_lower for kw in neutral_kws)
    has_pos = any(kw in hl_lower for kw in pos_kws)
    has_neg = any(kw in hl_lower for kw in neg_kws)
    
    if has_neg:
        return 'negative', score if label == 'negative' else max(score, 0.8)
    if has_pos:
        return 'positive', score if label == 'positive' else max(score, 0.8)
    if has_neutral:
        return 'neutral', 1.0
    return label, score

def analyze_company(t_code, c_name):
    # 1. yfinance 가격 수집
    start_dt = "2025-01-01"
    end_dt = "2026-04-30"
    
    stock_return = 0.0
    price_series = pd.DataFrame()
    stock_overheating_score = 0.0
    
    try:
        with st.spinner("주가 정보 및 변동성 분석 중..."):
            ticker = yf.Ticker(t_code)
            price_series = ticker.history(start=start_dt, end=end_dt)
            if not price_series.empty:
                close_start = price_series['Close'].iloc[0]
                close_end = price_series['Close'].iloc[-1]
                stock_return = (close_end - close_start) / close_start
                
                # 20일 이동평균 괴리율 계산
                moving_avg = price_series['Close'].rolling(window=20).mean().iloc[-1]
                current_price = price_series['Close'].iloc[-1]
                price_disparity = current_price / moving_avg if moving_avg > 0 else 1.0
                stock_overheating_score = float(np.clip((price_disparity - 1.0) / 0.6, 0.0, 1.0))
    except Exception as e:
        st.error(f"주가 분석 실패: {e}")
        
    # 2. 실시간 뉴스 크롤링 및 감성 분석
    headlines = []
    with st.spinner(f"네이버 뉴스 '{c_name}' 실시간 뉴스 수집 중..."):
        headlines = crawl_naver_news(c_name)
        
    sentiment_scores = []
    news_details = []
    if headlines:
        for hl in headlines:
            try:
                pred = classifier(hl)[0]
                label = pred['label'].lower()
                score = pred['score']
                
                # FIN-BERT 보정 필터 적용
                label, score = refine_sentiment(hl, label, score)
                
                if label == 'positive':
                    val = score
                elif label == 'negative':
                    val = -score
                else:
                    val = 0.0
                sentiment_scores.append(val)
                news_details.append({
                    "title": hl,
                    "label": label,
                    "score": score
                })
            except Exception:
                continue
                
    mean_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
    
    # 3. 미디어 관심도 계산
    hype_keywords = ['임상', '체결', 'fda', '인수', 'm&a', '특허', '기술이전']
    hype_keyword_count = 0
    if headlines:
        for hl in headlines:
            hl_lower = hl.lower()
            if any(kw in hl_lower for kw in hype_keywords):
                hype_keyword_count += 1
        media_attention_score = hype_keyword_count / len(headlines)
    else:
        media_attention_score = 0.5
        
    return stock_return, mean_sentiment, headlines, price_series, media_attention_score, stock_overheating_score, news_details

# -----------------------------------------------------------------------------
# 5. 사이드바 구성 (종목 코드 입력창)
# -----------------------------------------------------------------------------
st.sidebar.header(" 분석 종목 코드 입력")
raw_code = st.sidebar.text_input("6자리 종목 코드 입력", value="000250", placeholder="예: 000250 (삼천당제약)")

company_info, is_preset = get_company_info(raw_code, analysis_year)

# 기업명 표시
company_name_temp = company_info["name"] if company_info else f"기업_{raw_code}"
st.sidebar.markdown(f"** 기업명:** <span style='font-size: 15px; font-weight: bold; color: {'#38bdf8' if is_dark else '#0284c7'};'>{company_name_temp}</span>", unsafe_allow_html=True)

if company_info is None:
    company_info = {
        "name": f"기업_{raw_code}",
        "suffix": ".KQ",
        "revenue": 10e9,
        "rd_ratio": 8.0,
        "financing": 0,
        "impairment": 0.0,
        "ebitat": 0.0,
        "seqat": 0.5,
        "psr": 2.0,
        "listing_years": 5.0,
        "disclosures": 0,
        "rd_staff": 100,
        "phd_count": 20,
        "is_scraped": False
    }

clean_code = ''.join(filter(str.isdigit, raw_code))
company_name_display = company_info["name"]
ticker_code = clean_code + company_info["suffix"]
rd_staff = company_info.get("rd_staff", 100)
phd_count = company_info.get("phd_count", 20)
import re as _re_ylabel
_ym = _re_ylabel.search(r"(\d{4})년", analysis_year)
y_label = f"{_ym.group(1)}년" if _ym else "2024년"
_data_unavailable = company_info.get("data_unavailable", False)

# Initialize variables from company_info
init_revenue = float(company_info.get("revenue", 10e9))
init_rd_ratio = float(company_info.get("rd_ratio", 8.0))
init_financing = int(company_info.get("financing", 0))
init_impairment = float(company_info.get("impairment", 0.0))
init_ebitat = float(company_info.get("ebitat", 0.0))
init_seqat = float(company_info.get("seqat", 0.5))
init_psr = float(company_info.get("psr", 2.0))
init_listing_years = float(company_info.get("listing_years", 5.0))
init_disclosures = int(company_info.get("disclosures", 0))

st.markdown("---")
st.subheader(f"📊 {company_name_display} ({ticker_code}) - {y_label} 8대 핵심 지표")

if _data_unavailable:
    st.warning(f"선택하신 **{y_label}** 기준의 재무 데이터를 수집할 수 없습니다. 다른 연도를 선택해 주세요.")
    revenue_val = init_revenue
    rd_ratio = init_rd_ratio
    financing_count = init_financing
    impairment_val = init_impairment
    ebitat_val = init_ebitat
    seqat_val = init_seqat
    psr_val = init_psr
    listing_years = init_listing_years
    disclosures = init_disclosures
else:
    # Checkbox for simulation mode (only shown for presets, since non-presets have to use sliders)
    use_simulation = False
    if is_preset:
        use_simulation = st.checkbox("🔧 지표 직접 조정 (시뮬레이션 모드)", value=st.session_state.get("use_simulation", False))
        st.session_state["use_simulation"] = use_simulation
        
    if not is_preset or use_simulation:
        st.info("지표 값을 직접 조절하여 모델의 위험도 점수 변화를 시뮬레이션할 수 있습니다.")
        col_slide1, col_slide2 = st.columns(2)
        with col_slide1:
            revenue_val = col_slide1.slider(f"1. 매출액 (억원)", 0.0, 1000.0, float(init_revenue/1e8), step=1.0) * 1e8
            rd_ratio = col_slide1.slider("2. 매출 대비 R&D 비율 (%)", 0.0, 50.0, float(init_rd_ratio), step=0.1)
            financing_count = col_slide1.number_input("3. 3개년 유상증자 횟수 (회)", 0, 10, int(init_financing))
            impairment_val = col_slide1.slider(f"4. 자본잠식률 (%)", 0.0, 100.0, float(init_impairment), step=0.1)
        with col_slide2:
            ebitat_val = col_slide2.slider("5. 자본영업이익(EBITAT)", -0.5, 0.5, float(init_ebitat), step=0.01)
            seqat_val = col_slide2.slider("6. 자기자본비율(SEQAT)", 0.0, 1.0, float(init_seqat), step=0.01)
            psr_val = col_slide2.slider("7. PSR 비율 (고평가)", 0.0, 100.0, float(init_psr), step=0.1)
            listing_years = col_slide2.slider("8. 상장 경과년수 (년)", 0.0, 60.0, float(init_listing_years), step=0.5)
            disclosures = col_slide2.number_input("정정/미확정 공시 횟수", 0, 50, int(init_disclosures))
    else:
        revenue_val = init_revenue
        rd_ratio = init_rd_ratio
        financing_count = init_financing
        impairment_val = init_impairment
        ebitat_val = init_ebitat
        seqat_val = init_seqat
        psr_val = init_psr
        listing_years = init_listing_years
        disclosures = init_disclosures
        
        # Render beautiful grid
        bg_card = "#1e293b" if is_dark else "#ffffff"
        border_card = "#334155" if is_dark else "#e2e8f0"
        label_color = "#94a3b8" if is_dark else "#64748b"
        val_color = "#38bdf8" if is_dark else "#0284c7"
        
        cols_grid = st.columns(4)
        metrics_list = [
            ("1. 매출액", f"{revenue_val/1e8:,.1f} 억원", "연간 매출 규모"),
            ("2. 매출 대비 R&D 비율", f"{rd_ratio:.2f} %", "기술 투자 적극성"),
            ("3. 3개년 유상증자 횟수", f"{financing_count} 회", "자금 조달 빈도"),
            ("4. 자본잠식률", f"{impairment_val:.2f} %", "자본 잠식 리스크"),
            ("5. 자본영업이익 (EBITAT)", f"{ebitat_val:.4f}", "자산 대비 이익 효율"),
            ("6. 자기자본비율 (SEQAT)", f"{seqat_val:.4f}", "재무 안정성 지표"),
            ("7. PSR 비율", f"{psr_val:.2f} 배", "매출 대비 고평가"),
            ("8. 상장 경과년수", f"{listing_years:.1f} 년", "시장 안착 및 신뢰도")
        ]
        for i_m, (lbl, val_str, desc_str) in enumerate(metrics_list):
            c_idx = i_m % 4
            with cols_grid[c_idx]:
                st.markdown(f"""
                <div style="background-color: {bg_card}; border: 1px solid {border_card}; border-radius: 8px; padding: 12px; margin-bottom: 12px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                    <div style="font-size: 11px; color: {label_color}; font-weight: bold; margin-bottom: 3px;">{lbl}</div>
                    <div style="font-size: 16px; color: {val_color}; font-weight: bold; margin-bottom: 3px;">{val_str}</div>
                    <div style="font-size: 9.5px; color: {label_color};">{desc_str}</div>
                </div>
                """, unsafe_allow_html=True)
        # Display disclosures below the grid
        st.markdown(f"<p style='font-size:12px; color:{label_color}; text-align:right;'>※ 정정/미확정 공시 횟수: <b>{disclosures}회</b></p>", unsafe_allow_html=True)
st.markdown("---")
# Initialize session state variables for RAG Chatbot
if "current_code" not in st.session_state:
    st.session_state.current_code = clean_code
    st.session_state.current_year = analysis_year
    st.session_state.analysis_results = None
    st.session_state.messages = []

if st.session_state.current_code != clean_code or st.session_state.get("current_year", "") != analysis_year:
    st.session_state.current_code = clean_code
    st.session_state.current_year = analysis_year
    st.session_state.analysis_results = None
    st.session_state.messages = []

run_analysis = False
if st.sidebar.button(" Anti-Hype 분석 실행", type="primary"):
    clean_code_check = ''.join(filter(str.isdigit, raw_code))
    if len(clean_code_check) != 6:
        st.sidebar.error("올바른 6자리 종목 코드를 입력해 주세요.")
    else:
        run_analysis = True

if run_analysis:
    clean_code = ''.join(filter(str.isdigit, raw_code))
    company_name_display = company_info["name"]
    ticker_code = clean_code + company_info["suffix"]
    
    #  4단계 데이터 파이프라인 (Data Pipeline) 시각화 및 실행
    st.subheader(" 4단계 데이터 파이프라인 (Data Pipeline)")
    
    # Step 1: 수집 (Collect)
    with st.status("Step 1: 수집 (Collecting data from yfinance & Naver News)...", expanded=True) as status_step1:
        stock_ret, news_sent, news_list, price_data, media_attn, stock_overheat, news_details = analyze_company(ticker_code, company_name_display)
        status_step1.update(label="Step 1: 수집 완료 (주가 및 뉴스 수집 성공)", state="complete")
        
    # Step 2: 연산 (Compute)
    with st.status("Step 2: 연산 (Calculating Substance and Hype scores)...", expanded=True) as status_step2:
        combined_path = f"csv_combined/{clean_code}_combined.csv"
        if os.path.exists(combined_path):
            company_data, err = load_company_data_from_combined_csv(clean_code)
        else:
            company_data, err = risk_engine.fetch_and_analyze_company(clean_code)
        
        # 1. 벤치마크 피어 기업 목록
        peers = ["삼천당제약", "유한양행", "삼성바이오로직스", "한미약품", "셀트리온", "알테오젠", "리가켐바이오", "신라젠", "카나리아바이오", "박셀바이오", "고바이오랩"]
        
        # 피어 한글명 -> 종목코드 매핑 정보 (인코딩/이름 불일치 방지용)
        PEER_NAME_TO_CODE = {
            "삼천당제약": "000250",
            "유한양행": "000100",
            "삼성바이오로직스": "207940",
            "한미약품": "128940",
            "셀트리온": "068270",
            "알테오젠": "196170",
            "리가켐바이오": "141080",
            "신라젠": "215600",
            "카나리아바이오": "086890",
            "박셀바이오": "323990",
            "고바이오랩": "348150",
            "종근당홀딩스": "001630"
        }
        
        # 2. 데이터베이스에서 피어들 로드
        peer_rows = []
        import re
        year_match = re.search(r"(\d{4})년", analysis_year)
        year_str = year_match.group(1) if year_match else "2025"
        y_pref = f"{year_str}_"
        
        if not df_database.empty:
            for p_name in peers:
                # 현재 타겟 기업명과 겹치면 패스
                if p_name == company_name_display:
                    continue
                p_code = PEER_NAME_TO_CODE.get(p_name, "")
                match_p = df_database[df_database['clean_code'] == p_code] if p_code else pd.DataFrame()
                if not match_p.empty:
                    p_row = match_p.iloc[0]
                    
                    # 3개년 유증 계산
                    try:
                        curr_y = int(year_str)
                    except ValueError:
                        curr_y = 2025
                    
                    p_fin = 0
                    for y_offset in [0, 1, 2]:
                        y_check = curr_y - y_offset
                        if y_check >= 2021:
                            col_name = f'{y_check}_유상증자조달액(원)'
                            if col_name in p_row:
                                if float(p_row.get(col_name, 0)) > 0:
                                    p_fin += 1
                    
                    # PSR 계산
                    try:
                        close_val = float(p_row.get(f'{y_pref}Close', 0))
                        shares24 = float(p_row.get('sharesOutstanding', 0))
                        rev_val = float(p_row.get(f'{y_pref}매출액(원)', 0))
                        p_psr = (close_val * shares24) / max(1.0, rev_val)
                    except Exception:
                        p_psr = 1.0
                        
                    peer_rows.append({
                        "Name": p_name,
                        "Code": p_code,
                        "Revenue": float(p_row.get(f'{y_pref}매출액(원)', 0.0)),
                        "RD_Ratio": float(p_row.get(f'{y_pref}매출대비R&D비율(%)', 0.0)),
                        "Financing": p_fin,
                        "Impairment": float(p_row.get(f'{y_pref}자본잠식률(%)', 0.0)),
                        "EBITAT": float(p_row.get(f'{y_pref}EBITAT', 0.0)),
                        "SEQAT": float(p_row.get(f'{y_pref}SEQAT', 0.5)),
                        "PSR": p_psr,
                        "Listing_Years": float(p_row.get('years_after_listing', 5.0)) + (float(year_str) - 2024.0)
                    })
        
        # 3. 현재 분석 중인 타겟 기업 데이터 추가
        peer_rows.append({
            "Name": company_name_display,
            "Code": clean_code,
            "Revenue": revenue_val,
            "RD_Ratio": rd_ratio,
            "Financing": financing_count,
            "Impairment": impairment_val,
            "EBITAT": ebitat_val,
            "SEQAT": seqat_val,
            "PSR": psr_val,
            "Listing_Years": listing_years
        })
        
        df_plot = pd.DataFrame(peer_rows)
        
        # 4. 정규화 연산 (업계 표준 절대평가 벤치마크 스케일링)
        rev_log_plot = np.log10(df_plot["Revenue"].clip(lower=1e9))
        df_plot["score_log_sale"] = np.clip((rev_log_plot - 9.0) / 2.0, 0.0, 1.0)
        df_plot["score_rd"] = np.clip(df_plot["RD_Ratio"], 0.0, 20.0) / 20.0
        df_plot["score_financing"] = 1.0 - (df_plot["Financing"].clip(upper=3) / 3.0)
        df_plot["score_impairment"] = 1.0 - (np.clip(df_plot["Impairment"], 0.0, 50.0) / 50.0)
        df_plot["score_ebitat"] = np.clip((df_plot["EBITAT"] - (-0.1)) / 0.2, 0.0, 1.0)
        df_plot["score_seqat"] = np.clip((df_plot["SEQAT"] - 0.1) / 0.4, 0.0, 1.0)
        psr_log_plot = np.log10(df_plot["PSR"].clip(lower=1.0))
        df_plot["score_log_psr"] = np.clip(psr_log_plot / np.log10(30.0), 0.0, 1.0)
        df_plot["score_listing_years"] = np.clip(df_plot["Listing_Years"] / 15.0, 0.0, 1.0)
        
        # 최적 피처 컬럼명 추출
        feature_cols = [
            'feat_hype_volatility', 'feat_hype_volume_surge', 'feat_hype_momentum',
            'feat_hype_mdd', 'feat_hype_high_gap', 'feat_hype_LOG_PSR', 'feat_hype_LOG_PBR',
            'feat_fin_Capital_Impairment_Ratio', 'feat_fin_Debt_Ratio_Pct',
            'feat_fin_consecutive_op_loss_years', 'feat_fin_Paid_In_Capital_Increase',
            'feat_fin_3개년_유상증자_누적횟수', 'feat_fin_cb_bw_count',
            'feat_fin_insider_holdings_pct', 'feat_fin_Total_Revenue',
            'feat_fin_Total_Equity', 'feat_fin_Total_Assets', 'feat_fin_has_dividend',
            'feat_rd_R&D_Ratio_Pct', 'feat_rd_years_after_listing', 'feat_rd_is_tech_listing'
        ]
        
        hype_f_cols = [c for c in feature_cols if 'feat_hype_' in c]
        fin_f_cols = [c for c in feature_cols if 'feat_fin_' in c]
        rd_f_cols = [c for c in feature_cols if 'feat_rd_' in c]
        
        ensemble_probs = []
        hype_scores = []
        substance_scores = []
        rnd_scores = []
        target_feat_series = None
        target_f_to_i = None
        
        for idx_plot, row in df_plot.iterrows():
            c_name = row["Name"]
            c_code = row.get("Code", "")
            feat_series = pd.Series(0.5, index=feature_cols)
            
            match_db = df_database[df_database['clean_code'] == c_code] if (c_code and not df_database.empty) else pd.DataFrame()
            if not match_db.empty:
                code_val = match_db.iloc[0]['clean_code']
                if df_features_year is not None and code_val in df_features_year.index:
                    feat_series = df_features_year.loc[code_val].copy()
            
            # 사용자의 실시간 수정 슬라이더값 대치 (미등록 커스텀 기업인 경우에만 수행하여 DB 정합성 유지)
            if c_code == clean_code and not is_preset:
                feat_series['feat_fin_Total_Revenue'] = 1.0 - row["score_log_sale"]
                feat_series['feat_rd_R&D_Ratio_Pct'] = 1.0 - row["score_rd"]
                feat_series['feat_fin_3개년_유상증자_누적횟수'] = 1.0 - row["score_financing"]
                feat_series['feat_fin_Capital_Impairment_Ratio'] = 1.0 - row["score_impairment"]
                feat_series['feat_fin_Total_Revenue'] = 1.0 - row["score_ebitat"]
                feat_series['feat_fin_Total_Equity'] = 1.0 - row["score_seqat"]
                feat_series['feat_hype_LOG_PSR'] = row["score_log_psr"]
                feat_series['feat_rd_years_after_listing'] = 1.0 - row["score_listing_years"]
            
            if ensemble_model is not None:
                feat_df = pd.DataFrame([feat_series])[feature_cols]
                proba_result = ensemble_model.predict_proba(feat_df)
                if proba_result.shape[1] >= 2:
                    prob = proba_result[0, 1]
                else:
                    prob = proba_result[0, 0] if ensemble_model.classes_[0] == 1 else 1.0 - proba_result[0, 0]
                
                # SOTA 피처 중요도 기반 가중 리스크 점수
                imps = ensemble_model.feature_importances_
                # 단일 클래스 학습 시 균등 가중치 대체
                if sum(imps) == 0:
                    imps = np.ones(len(feature_cols)) / len(feature_cols)
                f_to_i = dict(zip(feature_cols, imps))
                
                sum_h_i = sum(f_to_i[c] for c in hype_f_cols)
                sum_f_i = sum(f_to_i[c] for c in fin_f_cols)
                sum_r_i = sum(f_to_i[c] for c in rd_f_cols)
                
                h_sc = feat_series[hype_f_cols].mul([f_to_i[c] for c in hype_f_cols]).sum() / (sum_h_i if sum_h_i > 0 else 1.0)
                f_sc = feat_series[fin_f_cols].mul([f_to_i[c] for c in fin_f_cols]).sum() / (sum_f_i if sum_f_i > 0 else 1.0)
                r_sc = feat_series[rd_f_cols].mul([f_to_i[c] for c in rd_f_cols]).sum() / (sum_r_i if sum_r_i > 0 else 1.0)
            else:
                prob = 0.5
                h_sc, f_sc, r_sc = 0.5, 0.5, 0.5
                
            ensemble_probs.append(prob)
            hype_scores.append(h_sc)
            substance_scores.append(f_sc)
            rnd_scores.append(r_sc)
            
            # 대상 기업의 피처 시리즈와 중요도 딕셔너리를 저장 (상세 분해용)
            if c_code == clean_code:
                target_feat_series = feat_series.copy()
                target_f_to_i = f_to_i.copy()
            
        df_plot["Hype_Index"] = ensemble_probs
        df_plot["Hype_Score"] = hype_scores
        df_plot["Substance_Score"] = substance_scores
        df_plot["Rnd_Score"] = rnd_scores
        
        target_row = df_plot[df_plot["Code"] == clean_code].iloc[0]
        substance_score = target_row["Substance_Score"]
        hype_score = target_row["Hype_Score"]
        rnd_score = target_row["Rnd_Score"]
        hype_index = target_row["Hype_Index"]
        
        status_step2.update(label=f"Step 2: 연산 완료 (Substance: {substance_score:.2f}, Hype: {hype_score:.2f})", state="complete")
        
    # Step 3: 팩트화 (Fact)
    with st.status("Step 3: 팩트화 (Saving the 5 core CSV files to the workspace directory)...", expanded=True) as status_step3:
        if not company_data:
            # Fallback data in case of fetch failures
            fallback_fin = pd.DataFrame([{
                "Year": datetime.now().strftime("%Y-%m-%d"),
                "Revenue": 0,
                "Operating_Income": 0,
                "Net_Income": 0,
                "R&D_Expense": 0,
                "R&D_Ratio_Pct": rd_ratio,
                "Common_Stock": 1000000000,
                "Total_Equity": 1000000000,
                "Total_Assets": 2000000000,
                "Total_Liabilities": 1000000000,
                "Capital_Impairment_Ratio": 0.0,
                "Debt_Ratio_Pct": 100.0,
                "Paid_In_Capital_Increase": 0,
                "Revenue_Growth_YoY_Pct": 0.0,
                "Status": "거래 정상"
            }])
            fallback_summary = {
                "code": clean_code,
                "name": company_name_display,
                "market": "KOSDAQ",
                "listing_date": "N/A",
                "is_tech_listing": True,
                "is_suspended": False,
                "suspension_date": "N/A",
                "cutoff_date": "N/A",
                "consecutive_low_rev_years": 0,
                "paid_in_count": 0,
                "paid_in_amount": 0,
                "insider_holdings_pct": 18.5,
                "final_status": "거래 정상"
            }
            company_data = {"summary": fallback_summary, "financials": fallback_fin}
            
        combined_path = f"csv_combined/{clean_code}_combined.csv"
        if os.path.exists(combined_path):
            saved_paths = []
        else:
            saved_paths = risk_engine.save_5_csv_files(company_data, output_dir="csv", code=clean_code)
        status_step3.update(label=f"Step 3: 팩트화 완료 (5대 CSV 확인 완료)", state="complete")
        
    # 핵심 판별 변수 계산
    consecutive_low_rev_years = 0
    paid_in_count = 0
    paid_in_amount = 0
    insider_holdings_pct = 18.5
    disclosures_val = 0
    
    if company_data:
        consecutive_low_rev_years = company_data["summary"].get("consecutive_low_rev_years", 0)
        paid_in_count = company_data["summary"].get("paid_in_count", 0)
        paid_in_amount = company_data["summary"].get("paid_in_amount", 0)
        insider_holdings_pct = company_data["summary"].get("insider_holdings_pct", 18.5)
        disclosures_val = company_data["summary"].get("disclosures", 0)
        
    if consecutive_low_rev_years >= 3:
        rev_status = " 위험"
        rev_color = "#ef4444"
        rev_desc = f"{consecutive_low_rev_years}년 연속 매출 30억 미만 (장기화 리스크)"
    elif consecutive_low_rev_years > 0:
        rev_status = " 주의"
        rev_color = "#eab308"
        rev_desc = f"{consecutive_low_rev_years}년 연속 매출 30억 미만"
    else:
        rev_status = " 정상"
        rev_color = "#22c55e"
        rev_desc = "매출 정상 발생 중 (30억 이상)"
        
    if disclosures_val >= 3:
        discl_status = " 위험"
        discl_color = "#ef4444"
        discl_desc = f"미확정/정정 공시 3개년 누적 {disclosures_val}회 (3회 이상 경고)"
    elif disclosures_val > 0:
        discl_status = " 주의"
        discl_color = "#eab308"
        discl_desc = f"미확정/정정 공시 3개년 누적 {disclosures_val}회"
    else:
        discl_status = " 정상"
        discl_color = "#22c55e"
        discl_desc = "정정 공시 빈도 양호"
        
    if paid_in_count >= 2 or paid_in_amount > 0:
        fin_status = " 위험" if paid_in_count >= 2 else " 주의"
        fin_color = "#ef4444" if paid_in_count >= 2 else "#eab308"
        fin_desc = f"최근 3년 유상증자 {paid_in_count}회, 총 {paid_in_amount:,}원 조달"
    else:
        fin_status = " 정상"
        fin_color = "#22c55e"
        fin_desc = "영업활동 중심의 자금 구조 유지"

    if insider_holdings_pct < 20:
        insider_status = " 주의"
        insider_color = "#eab308"
        insider_desc = f"대주주 지분율 {insider_holdings_pct:.1f}%로 경영권 취약 및 지분 이탈 우려 (20% 미만)"
    else:
        insider_status = " 정상"
        insider_color = "#22c55e"
        insider_desc = f"대주주 지분율 {insider_holdings_pct:.1f}%로 지분율 양호"

    # Step 4: 요약 (AI 상세 리포트 생성)
    with st.status(f"Step 4: 요약 ({ai_provider} 상세 리포트 생성 중)...", expanded=True) as status_step4:
        llm_report = ""
        if active_api_key:
            report_prompt = f"""당신은 제약/바이오 기업의 DART 공시와 재무 상태를 심도 있게 분석하여 투자자에게 실체적 가치와 시장 과열(Hype) 리스크를 평가해 주는 전문 금융 분석가(Anti-Hype Engine AI)입니다.
            
            다음 기업에 대한 상세 리포트를 한글로 작성해 주세요.
            
            [분석 기업 정보]
            - 기업명: {company_name_display}
            - 종목코드: {clean_code}
            - 상장시장: {company_info.get("market", "KOSDAQ")}
            
            [분석 결과 스코어 (0~100점, 높을수록 위험)]
            - HYPE 리스크 점수: {hype_score*100:.1f}점
            - 재무 리스크 점수: {substance_score*100:.1f}점
            - R&D 리스크 점수: {rnd_score*100:.1f}점
            - 종합 붕괴 위험도: {hype_index*100:.1f}%

            [AI 리스크 피처 해석 가이드라인 (반드시 준수)]
            1. R&D 리스크 점수 (높을수록 위험): 매출액 대비 R&D 비율이 낮거나(R&D 투자 부족), 상장 경과년수가 짧거나, 기술특례상장 기업인 경우 리스크 점수가 높게 나옵니다.
               - 주의: R&D 비율 자체는 높을수록 R&D 투자를 적극적으로 잘 하고 있는 우수 신호(낮은 리스크)이며, 낮을수록 기술 공동화 우려가 큰 위험 신호(높은 리스크)입니다. R&D 비율이 높은데도 R&D 리스크 점수가 높은 경우는 짧은 상장 연수나 기술특례상장 요인 때문입니다. R&D 투자를 많이 하는 행위를 "R&D 공동화" 또는 "부정적 요인"으로 설명하는 인과관계 오류를 절대 범하지 마십시오.
            2. 재무 리스크 점수 (높을수록 위험): 자본잠식률이나 부채비율이 높을수록, 연속 영업손실 연수가 길수록 점수가 높게 나옵니다.
            3. HYPE 리스크 점수 (높을수록 위험): 주가 변동성, 거래량 급증 비율, PSR/PBR 등이 높을수록 점수가 높게 나옵니다.
            
            [핵심 리스크 임계값 상태]
            1. 매출 0원 기간: {rev_status} ({rev_desc})
            2. 정정 공시 빈도: {discl_status} ({discl_desc})
            3. 재무활동 조달 편중: {fin_status} ({fin_desc})
            4. 대주주 이탈: {insider_status} ({insider_desc})
            
            [최근 뉴스 헤드라인]
            {chr(10).join(f'- {h}' for h in news_list[:5]) if news_list else '없음'}
            
            [리포트 작성 가이드라인]
            1. 전문적이고 객관적인 톤앤매너: 감정적이지 않고, 냉철하고 예방적인 투자 리스크 진단을 제공하십시오.
            2. 구조화된 평문(Plain Text) 형식:
               - 별표(**)나 글머리 기호(- 등) 같은 마크다운 서식을 절대 사용하지 마십시오.
               - 각 섹션 번호와 제목은 다음과 같이 순수한 평문(Plain Text) 텍스트로만 시작해 주십시오. (예: "1. 종합 진단 요약", "5. 투자자 최종 대응 가이드라인" 등)
               - 각 섹션 구성:
                 1. 종합 진단 요약: 기업의 종합적인 리스크 수준과 상태를 요약 (1-2문장)
                 2. 시장 과열(Hype) 리스크 분석: 주가 변동성, 최근 뉴스 호재 여부 및 PSR 수준 평가
                 3. 재무 건전성(Substance) 및 R&D 리스크 분석: 자산대비영업이익(EBITAT), R&D 비율 및 매출액 미달 위험성 분석
                 4. 핵심 우려 요인 및 경고 신호: 유상증자, CB/BW 발행, 공시 번복(정정), 대주주 지분율 등 4대 판별 변수 위주로 지적
                 5. 투자자 최종 대응 가이드라인: 보수적 관점에서의 대응 전략 제시
            3. 마크다운 금지: 답변 내용 전체에 마크다운 문법(예: **, *, -, # 등)을 절대로 사용하지 마십시오. 오직 일반 줄바꿈과 공백으로만 가독성을 조절해 주십시오.
            4. 표준 한국어 준수: 자연스러운 표준 한국어(한글)로만 리포트를 작성하십시오. 모든 어휘는 문법에 맞고 정확한 표준 한글(예: '결정' 등)로만 표현해야 합니다.
            5. 한자 절대 금지: 답변 내용 전체에 한자(예: 株式, 負債, 投資, 決定 등)를 단 한 글자도 사용하지 마십시오. 모든 단어는 반드시 한자 없이 순수한 한글로만 작성해야 합니다.
            6. 길이: 가독성을 위해 각 섹션별로 핵심적인 내용만 2~3줄씩 요약하여 명확하게 작성해 주세요. 불필요하게 장황한 서술은 피하십시오.
            """
            messages_payload = [
                {"role": "system", "content": "당신은 제약/바이오 리스크 전문 분석가입니다."},
                {"role": "user", "content": report_prompt}
            ]
            report_text, err = call_llm_api(
                provider=ai_provider,
                api_key=active_api_key,
                model=active_model,
                messages=messages_payload,
                temperature=0.3
            )
            if err:
                llm_report = f"⚠️ {ai_provider} API 호출 중 오류가 발생했습니다: {err}"
            else:
                report_text = report_text.replace("quyết정", "결정").replace("quyết", "결정")
                llm_report = report_text
        else:
            llm_report = f"⚠️ 사이드바에 {ai_provider} API Key를 입력하시면, AI가 공시 및 재무 데이터를 기반으로 작성한 한글 상세 진단 리포트를 이곳에 실시간으로 생성해 드립니다."
        status_step4.update(label="Step 4: 요약 완료 (상세 리포트 생성 성공)", state="complete")
        
    st.session_state.analysis_results = {
        "stock_ret": stock_ret,
        "news_sent": news_sent,
        "news_list": news_list,
        "news_details": news_details,
        "price_data": price_data,
        "media_attn": media_attn,
        "stock_overheat": stock_overheat,
        "substance_score": substance_score,
        "hype_score": hype_score,
        "rnd_score": rnd_score,
        "hype_index": hype_index,
        "company_data": company_data,
        "target_feat_series": target_feat_series,
        "target_f_to_i": target_f_to_i,
        "hype_f_cols": hype_f_cols,
        "fin_f_cols": fin_f_cols,
        "rd_f_cols": rd_f_cols,
        "df_plot": df_plot,
        "llm_report": llm_report,
        "rev_status": rev_status,
        "rev_desc": rev_desc,
        "rev_color": rev_color,
        "discl_status": discl_status,
        "discl_desc": discl_desc,
        "discl_color": discl_color,
        "fin_status": fin_status,
        "fin_desc": fin_desc,
        "fin_color": fin_color,
        "insider_status": insider_status,
        "insider_desc": insider_desc,
        "insider_color": insider_color
    }
    st.rerun()

# -------------------------------------------------------------------------
# 결과 대시보드 카드 및 챗봇 렌더링
# -------------------------------------------------------------------------
def _build_unified_breakdown_html(feat_series, f_to_i, hype_f_cols, fin_f_cols, rd_f_cols):
    bg_color = "#1e293b" if is_dark else "#f8fafc"
    text_color = "#f8fafc" if is_dark else "#0f172a"
    border_color = "#334155" if is_dark else "#e2e8f0"
    header_bg = "#0f172a" if is_dark else "#1e3d59"
    weight_color = "#94a3b8" if is_dark else "#64748b"
    contrib_color = "#38bdf8" if is_dark else "#0284c7"
    
    html = f"""
    <div style="margin-top:15px; margin-bottom:15px;">
        <table style="width:100%; border-collapse:collapse; font-size:12px; border:1px solid {border_color}; margin-bottom:15px; background-color:{bg_color}; color:{text_color};">
            <thead>
                <tr style="background-color:{header_bg}; color:white; font-weight:bold;">
                    <th style="padding:8px 10px; text-align:left; border-right:1px solid {border_color}; border-bottom:1px solid {border_color};">피처 설명 (Feature)</th>
                    <th style="padding:8px 10px; text-align:center; border-right:1px solid {border_color}; border-bottom:1px solid {border_color};">리스크 분류 (Category)</th>
                    <th style="padding:8px 10px; text-align:center; border-right:1px solid {border_color}; border-bottom:1px solid {border_color};">정규화 값 (Value)</th>
                    <th style="padding:8px 10px; text-align:center; border-right:1px solid {border_color}; border-bottom:1px solid {border_color};">RF 모델 중요도 (Weight)</th>
                    <th style="padding:8px 10px; text-align:center; border-bottom:1px solid {border_color};">종합 붕괴 기여도 (Contribution)</th>
                </tr>
            </thead>
            <tbody>
    """
    
    feature_names_kr = {
        'feat_hype_volatility': '주가 변동성 (20일 변동성)',
        'feat_hype_volume_surge': '거래량 급증 비율',
        'feat_hype_momentum': '주가 모멘텀 (1개월 수익률)',
        'feat_hype_mdd': '최대 낙폭 (MDD)',
        'feat_hype_high_gap': '고점 대비 괴리율',
        'feat_hype_LOG_PSR': '로그 PSR 비율',
        'feat_hype_LOG_PBR': '로그 PBR 비율',
        'feat_fin_Capital_Impairment_Ratio': '자본잠식률 (%)',
        'feat_fin_Debt_Ratio_Pct': '부채비율 (%)',
        'feat_fin_consecutive_op_loss_years': '연속 영업적자 연수',
        'feat_fin_Paid_In_Capital_Increase': '유상증자 조달액',
        'feat_fin_3개년_유상증자_누적횟수': '3개년 유상증자 횟수',
        'feat_fin_cb_bw_count': 'CB/BW 발행 건수',
        'feat_fin_insider_holdings_pct': '대주주 지분율 (%)',
        'feat_fin_Total_Revenue': '매출액 규모',
        'feat_fin_Total_Equity': '자기자본 규모',
        'feat_fin_Total_Assets': '자산총계 규모',
        'feat_fin_has_dividend': '배당 지급 여부',
        'feat_rd_R&D_Ratio_Pct': '매출액 대비 R&D 비율 (%)',
        'feat_rd_years_after_listing': '상장 경과년수',
        'feat_rd_is_tech_listing': '기술특례상장 여부'
    }

    all_cols = list(feat_series.index)
    total_weighted_sum = sum(float(feat_series.get(c, 0.0)) * float(f_to_i.get(c, 0.0)) for c in all_cols)
    
    rows_data = []
    for c in all_cols:
        val = float(feat_series.get(c, 0.0))
        weight = float(f_to_i.get(c, 0.0))
        contrib = val * weight
        contrib_pct = (contrib / total_weighted_sum * 100) if total_weighted_sum > 0 else 0.0
        
        if c in hype_f_cols:
            category = "HYPE (시장 과열)"
            cat_color = "#f59e0b"
        elif c in fin_f_cols:
            category = "Financial (재무 부실)"
            cat_color = "#3b82f6"
        elif c in rd_f_cols:
            category = "R&D (기술 공동화)"
            cat_color = "#8b5cf6"
        else:
            category = "기타"
            cat_color = "#64748b"
            
        name_kr = feature_names_kr.get(c, c)
        rows_data.append({
            'col': c,
            'name_kr': name_kr,
            'category': category,
            'cat_color': cat_color,
            'val': val,
            'weight': weight,
            'contrib_pct': contrib_pct
        })
        
    # Sort rows by contrib_pct descending
    rows_data = sorted(rows_data, key=lambda r: r['contrib_pct'], reverse=True)
    
    for r in rows_data:
        val_color = "#ef4444" if r['val'] >= 0.7 else "#f59e0b" if r['val'] >= 0.4 else "#22c55e"
        
        html += f"""
                <tr style="border-bottom:1px solid {border_color};">
                    <td style="padding:6px 10px; border-right:1px solid {border_color}; font-weight:500;">{r['name_kr']} <span style="font-size:10px; color:{weight_color}; font-weight:normal;">({r['col']})</span></td>
                    <td style="padding:6px 10px; text-align:center; border-right:1px solid {border_color}; font-weight:bold; color:{r['cat_color']};">{r['category']}</td>
                    <td style="padding:6px 10px; text-align:center; border-right:1px solid {border_color}; font-weight:bold; color:{val_color};">{r['val']:.4f}</td>
                    <td style="padding:6px 10px; text-align:center; border-right:1px solid {border_color}; color:{weight_color};">{r['weight']*100:.2f}%</td>
                    <td style="padding:6px 10px; text-align:center; font-weight:bold; color:{contrib_color};">{r['contrib_pct']:.2f}%</td>
                </tr>
        """
        
    html += """
            </tbody>
        </table>
    </div>
    """
    return clean_html(html)

def _axis_grade(score, axis_type):
    s = score * 100
    if axis_type == 'hype':
        if s < 20:
            return ("안정", f"주가 과열 징후가 거의 없는 안정적 가격대(HYPE {s:.1f}점)")
        elif s < 40:
            return ("양호", f"주가 과열도가 낮은 편이나 일부 변동성이 존재(HYPE {s:.1f}점)")
        elif s < 65:
            return ("주의", f"시장 과열 신호가 감지되며 단기 조정 가능성이 있는 상태(HYPE {s:.1f}점)")
        else:
            return ("위험", f"주가가 실체 대비 과도하게 고평가된 거품 구간에 진입(HYPE {s:.1f}점)")
    elif axis_type == 'fin':
        if s < 30:
            return ("안정", f"재무 건전성이 견고하고 자기자본이 안정적(재무 {s:.1f}점)")
        elif s < 55:
            return ("양호", f"재무 구조가 보통 수준이나 일부 취약 요인이 존재(재무 {s:.1f}점)")
        elif s < 75:
            return ("주의", f"재무 체력이 약화되고 있어 주의가 필요한 구간(재무 {s:.1f}점)")
        else:
            return ("위험", f"재무 부실 징후가 심각하며 자본잠식/유동성 위기 경계(재무 {s:.1f}점)")
    else:  # rd
        if s < 30:
            return ("안정", f"R&D 투자와 기술 인프라가 탄탄하게 유지(R&D {s:.1f}점)")
        elif s < 55:
            return ("양호", f"R&D 수준이 보통이나 추가 기술 투자가 요구(R&D {s:.1f}점)")
        elif s < 75:
            return ("주의", f"R&D 투자가 부족하여 기술 경쟁력 정체 우려(R&D {s:.1f}점)")
        else:
            return ("위험", f"R&D 기능이 사실상 공동화되어 파이프라인 실체가 부재(R&D {s:.1f}점)")

tab1, tab2 = st.tabs([" 개별 기업 진단 리포트", " 2D 시계열 궤적 시각화"])

with tab1:
    if st.session_state.analysis_results is None:
        st.info(" 사이드바에서 기업 코드를 선택하고 **'Anti-Hype 분석 실행'** 버튼을 클릭하여 리스크 분석 및 AI 대화를 시작하세요.")
    else:
        res = st.session_state.analysis_results
        stock_ret = res["stock_ret"]
        news_sent = res["news_sent"]
        news_list = res["news_list"]
        news_details = res.get("news_details", [])
        price_data = res["price_data"]
        media_attn = res["media_attn"]
        stock_overheat = res["stock_overheat"]
        substance_score = res["substance_score"]
        hype_score = res["hype_score"]
        rnd_score = res["rnd_score"]
        hype_index = res["hype_index"]
        company_data = res["company_data"]
        target_feat_series = res["target_feat_series"]
        target_f_to_i = res["target_f_to_i"]
        hype_f_cols = res["hype_f_cols"]
        fin_f_cols = res["fin_f_cols"]
        rd_f_cols = res["rd_f_cols"]
        df_plot = res["df_plot"]
        
        llm_report = res.get("llm_report", "")
        rev_status = res.get("rev_status", "")
        rev_desc = res.get("rev_desc", "")
        rev_color = res.get("rev_color", "")
        discl_status = res.get("discl_status", "")
        discl_desc = res.get("discl_desc", "")
        discl_color = res.get("discl_color", "")
        fin_status = res.get("fin_status", "")
        fin_desc = res.get("fin_desc", "")
        fin_color = res.get("fin_color", "")
        insider_status = res.get("insider_status", "")
        insider_desc = res.get("insider_desc", "")
        insider_color = res.get("insider_color", "")

        st.subheader(f" {company_name_display} ({ticker_code}) 분석 결과")
        
        col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
        with col_c2:
            st.markdown(f"""
            <div class="report-card" style="text-align: center; border: 2px solid {'#ef4444' if hype_index > 0.6 else '#f59e0b' if hype_index > 0.4 else '#22c55e'}; background-color: {'#1e293b' if is_dark else '#ffffff'}; box-shadow: 0 4px 10px rgba(0,0,0,0.06); padding: 30px; border-radius: 12px; margin-bottom: 20px;">
                <div class="metric-label" style="font-size: 16px; font-weight: bold; color: {'#94a3b8' if is_dark else '#64748b'};">종합 붕괴 위험도 (Collapse Risk)</div>
                <div class="metric-value" style="font-size: 64px; font-weight: 800; color: {'#ef4444' if hype_index > 0.6 else '#f59e0b' if hype_index > 0.4 else '#22c55e'}; margin-top: 10px; margin-bottom: 5px;">{hype_index*100:.1f}%</div>
                <div style="font-size: 13px; color: {'#94a3b8' if is_dark else '#64748b'}; font-weight: 500;">
                    위험 상태: {'위험 (경고)' if hype_index > 0.6 else '주의 (관찰)' if hype_index > 0.4 else '정상 (양호)'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # 종합 진단 의견 alert box
        hype_grade, hype_desc_part = _axis_grade(hype_score, 'hype')
        fin_grade, fin_desc_part = _axis_grade(substance_score, 'fin')
        rd_grade, rd_desc_part = _axis_grade(rnd_score, 'rd')

        danger_count = sum(1 for g in [hype_grade, fin_grade, rd_grade] if g == "위험")
        caution_count = sum(1 for g in [hype_grade, fin_grade, rd_grade] if g == "주의")

        if danger_count >= 3:
            status_text = "위험군: 3대 리스크 극대화 (상장유지 위기 경보)"
            box_class = "warning-box"
        elif danger_count >= 2:
            status_text = "위험군: 복합 리스크 경고"
            box_class = "warning-box"
        elif danger_count >= 1:
            status_text = "주의군: 부분 리스크 감지"
            box_class = "warning-box"
        elif caution_count >= 2:
            status_text = "관찰군: 다수 축 주의 필요"
            box_class = "fair-box"
        elif caution_count >= 1:
            status_text = "관찰군: 일부 지표 주의"
            box_class = "fair-box"
        else:
            status_text = "양호군: 적정 가치 기업 (안정 투자군)"
            box_class = "undervalued-box"

        diag_desc = (
            f"본 기업({company_name_display})의 축별 진단 결과: "
            f"① 시장 과열({hype_grade}): {hype_desc_part}. "
            f"② 재무 건전성({fin_grade}): {fin_desc_part}. "
            f"③ R&D/기술({rd_grade}): {rd_desc_part}."
        )

        st.markdown(f'<div class="{box_class}"><strong>{status_text} (종합 붕괴 확률: {hype_index*100:.1f}%)</strong><br>{diag_desc}</div>', unsafe_allow_html=True)

        # 📰 실시간 뉴스 및 언론 감성 분석 (KR-FinBert-SC)
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(" 📰 실시간 뉴스 및 언론 감성 분석 (KR-FinBert-SC)")
        
        total_news = len(news_details)
        if total_news > 0:
            # Normalize labels for calculations
            norm_details = []
            for d in news_details:
                lbl = d["label"].lower()
                if lbl == 'label_0':
                    lbl = 'negative'
                elif lbl == 'label_1':
                    lbl = 'neutral'
                elif lbl == 'label_2':
                    lbl = 'positive'
                norm_details.append(lbl)
                
            pos_count = sum(1 for l in norm_details if l == "positive")
            neg_count = sum(1 for l in norm_details if l == "negative")
            neu_count = sum(1 for l in norm_details if l == "neutral")
            
            pos_pct = pos_count / total_news * 100
            neg_pct = neg_count / total_news * 100
            neu_pct = neu_count / total_news * 100
            
            # 3 columns for metrics
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; border-left: 5px solid #10b981; background-color: {'#1e293b' if is_dark else '#f8fafc'}; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <div style="font-size: 14px; font-weight: bold; color: {'#94a3b8' if is_dark else '#64748b'};">긍정 기사 비율</div>
                    <div style="font-size: 24px; font-weight: 800; color: #10b981; margin-top: 5px;">{pos_pct:.1f}%</div>
                    <div style="font-size: 11px; color: {'#64748b' if is_dark else '#94a3b8'};">{pos_count}건 / 전체 {total_news}건</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; border-left: 5px solid #6b7280; background-color: {'#1e293b' if is_dark else '#f8fafc'}; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <div style="font-size: 14px; font-weight: bold; color: {'#94a3b8' if is_dark else '#64748b'};">중립 기사 비율</div>
                    <div style="font-size: 24px; font-weight: 800; color: #6b7280; margin-top: 5px;">{neu_pct:.1f}%</div>
                    <div style="font-size: 11px; color: {'#64748b' if is_dark else '#94a3b8'};">{neu_count}건 / 전체 {total_news}건</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s3:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; border-left: 5px solid #ef4444; background-color: {'#1e293b' if is_dark else '#f8fafc'}; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                    <div style="font-size: 14px; font-weight: bold; color: {'#94a3b8' if is_dark else '#64748b'};">부정 기사 비율</div>
                    <div style="font-size: 24px; font-weight: 800; color: #ef4444; margin-top: 5px;">{neg_pct:.1f}%</div>
                    <div style="font-size: 11px; color: {'#64748b' if is_dark else '#94a3b8'};">{neg_count}건 / 전체 {total_news}건</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Stacked progress bar
            st.markdown(f"""
            <div style="margin-top: 15px; margin-bottom: 5px; font-size: 12px; font-weight: bold; color: {'#94a3b8' if is_dark else '#64748b'};">언론 긍부정 분포 (Sentiment Distribution)</div>
            <div style="display: flex; height: 16px; width: 100%; border-radius: 8px; overflow: hidden; margin-bottom: 15px; background-color: {'#334155' if is_dark else '#e2e8f0'};">
                <div style="width: {pos_pct}%; background-color: #10b981;" title="긍정 {pos_pct:.1f}%"></div>
                <div style="width: {neu_pct}%; background-color: #6b7280;" title="중립 {neu_pct:.1f}%"></div>
                <div style="width: {neg_pct}%; background-color: #ef4444;" title="부정 {neg_pct:.1f}%"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show crawled news list inside an expander — native Streamlit (no raw HTML table)
            with st.expander("📰 기사별 상세 감성 라벨 및 신뢰도 보기 (KR-FinBert-SC)", expanded=False):
                # Header row
                h1, h2, h3 = st.columns([6, 1.2, 1])
                h1.markdown(f"<span style='font-size:11px; font-weight:bold; color:{'#94a3b8' if is_dark else '#64748b'};'>헤드라인</span>", unsafe_allow_html=True)
                h2.markdown(f"<span style='font-size:11px; font-weight:bold; color:{'#94a3b8' if is_dark else '#64748b'};'>감성 분류</span>", unsafe_allow_html=True)
                h3.markdown(f"<span style='font-size:11px; font-weight:bold; color:{'#94a3b8' if is_dark else '#64748b'};'>신뢰도</span>", unsafe_allow_html=True)
                st.divider()

                for detail in news_details:
                    conf = detail["score"] * 100
                    det_lbl = detail["label"].lower()
                    if det_lbl == 'label_0':
                        det_lbl = 'negative'
                    elif det_lbl == 'label_1':
                        det_lbl = 'neutral'
                    elif det_lbl == 'label_2':
                        det_lbl = 'positive'

                    if det_lbl == "positive":
                        lbl_color = "#10b981"
                        lbl_bg = "rgba(16,185,129,0.15)"
                        lbl_text = "긍정"
                    elif det_lbl == "negative":
                        lbl_color = "#ef4444"
                        lbl_bg = "rgba(239,68,68,0.15)"
                        lbl_text = "부정"
                    else:
                        lbl_color = "#6b7280"
                        lbl_bg = "rgba(107,114,128,0.15)"
                        lbl_text = "중립"

                    c1, c2, c3 = st.columns([6, 1.2, 1])
                    c1.markdown(
                        f"<span style='font-size:12px; color:{'#f1f5f9' if is_dark else '#0f172a'};'>{detail['title']}</span>",
                        unsafe_allow_html=True
                    )
                    c2.markdown(
                        f"<span style='display:inline-block; padding:2px 8px; border-radius:4px; "
                        f"font-weight:bold; font-size:11px; color:{lbl_color}; "
                        f"background-color:{lbl_bg}; border:1px solid {lbl_color}44;'>"
                        f"{lbl_text}</span>",
                        unsafe_allow_html=True
                    )
                    c3.markdown(
                        f"<span style='font-size:12px; color:{'#94a3b8' if is_dark else '#64748b'};'>{conf:.1f}%</span>",
                        unsafe_allow_html=True
                    )
        else:
            st.info("수집된 실시간 뉴스 헤드라인이 없습니다.")

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. LLM 상세 진단 리포트 즉시 노출
        import re as _re_pt
        _pt_m = _re_pt.search(r"(\d{4})년", analysis_year)
        _pt_year = int(_pt_m.group(1)) if _pt_m else 2024
        if _pt_year >= 2025:
            predict_target_years = f"향후 1년 ~ 2년 내 ({_pt_year+2}년 ~ {_pt_year+3}년) 미래 대폭락 리스크"
        else:
            predict_target_years = f"과거 ({_pt_year+1}년 ~ {_pt_year+2}년) 대폭락 리스크"
        st.subheader(f" {predict_target_years} 종합 리스크 진단 리포트 (by OpenAI)")
        st.markdown(llm_report)

        if target_feat_series is not None and target_f_to_i is not None:
            # Build breakdown HTML tables inside expander
            with st.expander(" 3대 리스크 점수 피처별 상세 분해 (Feature-Level Breakdown)", expanded=False):
                st.markdown("각 피처의 정규화 값과 Random Forest 모델의 가중치를 곱하여 산출한 종합 붕괴 위험도 기여도 테이블입니다. **종합 붕괴 기여도(%)의 합은 100%**이며, 상위에 노출된 피처일수록 현재 종합 붕괴 위험도를 결정짓는 핵심 드라이버입니다.")
                st.markdown(_build_unified_breakdown_html(target_feat_series, target_f_to_i, hype_f_cols, fin_f_cols, rd_f_cols), unsafe_allow_html=True)

        with st.expander("️ 핵심 위험 변수 탐지 (Core Risk Parameters)", expanded=False):
            table_bg = "#1e293b" if is_dark else "#f8fafc"
            header_bg = "#0f172a" if is_dark else "#1e3d59"
            border_color = "#334155" if is_dark else "#e2e8f0"
            text_color = "#f8fafc" if is_dark else "#0f172a"
            
            html_table = f"""
            <table style="width:100%; border-collapse:collapse; font-size:14px; border: 1px solid {border_color}; margin-top:10px; margin-bottom:20px;">
                <thead>
                    <tr style="background-color:{header_bg}; color:white; font-weight:bold;">
                        <th style="padding:10px; text-align:left; border: 1px solid {border_color};">위험 판별 변수</th>
                        <th style="padding:10px; text-align:center; border: 1px solid {border_color}; width:100px;">상태</th>
                        <th style="padding:10px; text-align:left; border: 1px solid {border_color};">상세 진단 결과</th>
                    </tr>
                </thead>
                <tbody style="background-color:{table_bg}; color:{text_color};">
                    <tr style="border-bottom: 1px solid {border_color};">
                        <td style="padding:10px; font-weight:bold; border: 1px solid {border_color};">1. 매출 0원 기간</td>
                        <td style="padding:10px; text-align:center; color:{rev_color}; font-weight:bold; border: 1px solid {border_color};">{rev_status}</td>
                        <td style="padding:10px; border: 1px solid {border_color};">{rev_desc}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid {border_color};">
                        <td style="padding:10px; font-weight:bold; border: 1px solid {border_color};">2. 정정 공시 빈도</td>
                        <td style="padding:10px; text-align:center; color:{discl_color}; font-weight:bold; border: 1px solid {border_color};">{discl_status}</td>
                        <td style="padding:10px; border: 1px solid {border_color};">{discl_desc}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid {border_color};">
                        <td style="padding:10px; font-weight:bold; border: 1px solid {border_color};">3. 재무활동 조달 편중</td>
                        <td style="padding:10px; text-align:center; color:{fin_color}; font-weight:bold; border: 1px solid {border_color};">{fin_status}</td>
                        <td style="padding:10px; border: 1px solid {border_color};">{fin_desc}</td>
                    </tr>
                    <tr>
                        <td style="padding:10px; font-weight:bold; border: 1px solid {border_color};">4. 대주주 이탈 / 블록딜 매도</td>
                        <td style="padding:10px; text-align:center; color:{insider_color}; font-weight:bold; border: 1px solid {border_color};">{insider_status}</td>
                        <td style="padding:10px; border: 1px solid {border_color};">{insider_desc}</td>
                    </tr>
                </tbody>
            </table>
            """
            st.markdown(clean_html(html_table), unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader(" AI 대화형 실체 분석 및 리스크 진단 (RAG Chatbot)")
        st.markdown("DART 공시 데이터와 재무제표 벤치마크 데이터를 바탕으로 AI 분석가와 기업 리스크에 대해 질문하고 답을 얻을 수 있습니다.")
        
        if not active_api_key:
            st.warning(f"⚠️ AI 챗봇을 활성화하려면 사이드바에서 **{ai_provider} API Key**를 입력해 주세요. ({active_model} 모델이 실시간 연동됩니다)")
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            st.markdown("<p style='font-size:13px; color:#888;'> 자주 묻는 질문 (추천 질문):</p>", unsafe_allow_html=True)
            col_q1, col_q2 = st.columns(2)
            col_q3, col_q4 = st.columns(2)
            
            quick_query = None
            with col_q1:
                if st.button(" 매출 추이 및 영업적자 리스크 분석", use_container_width=True):
                    quick_query = "이 기업의 매출 규모와 연속 영업적자 발생 현황에 대해 재무제표를 바탕으로 자세히 분석해 줘."
            with col_q2:
                if st.button(" R&D 투자 및 기술 파이프라인 분석", use_container_width=True):
                    quick_query = "이 기업의 매출 대비 R&D 비율과 기술 파이프라인의 실체를 평가해 줘."
            with col_q3:
                if st.button("️ 유증 / CB / BW 발행과 자금 조달 리스크", use_container_width=True):
                    quick_query = "최근 3개년 유상증자 횟수와 CB/BW 발행 건수를 파악하여, 시장 조달 연명 비중이나 잠재 오버행 리스크를 분석해 줘."
            with col_q4:
                if st.button(" 대주주 지분율 및 종합 붕괴 리스크", use_container_width=True):
                    quick_query = "최대주주 지분율과 종합 붕괴 위험도를 토대로, 투자자가 가장 경계해야 할 핵심 위험 요소를 짚어줘."
    
            user_input = st.chat_input("기업의 재무, R&D, 공시 리스크에 대해 질문하세요...")
            if quick_query:
                user_input = quick_query
                
            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                csv_raw_text = ""
                combined_path = f"csv_combined/{clean_code}_combined.csv"
                if os.path.exists(combined_path):
                    for enc in ["utf-8-sig", "cp949", "utf-8"]:
                        try:
                            with open(combined_path, "r", encoding=enc) as f:
                                csv_raw_text = f.read()
                            break
                        except Exception:
                            continue
                
                system_prompt = f"""당신은 제약/바이오 기업의 DART 공시와 재무 상태를 심도 있게 분석하여 투자자에게 실체적 가치와 시장 과열(Hype) 리스크를 평가해 주는 전문 금융 분석가(Anti-Hype Engine AI)입니다.
    
    [분석 기업 정보]
    - 기업명: {company_name_display}
    - 종목코드: {clean_code}
    - 상장시장: {company_info.get("market", "KOSDAQ")}
    
    [분석 결과 스코어 (0~100점, 높을수록 위험)]
    - HYPE 리스크 점수: {hype_score*100:.1f}점
    - 재무 리스크 점수: {substance_score*100:.1f}점
    - R&D 리스크 점수: {rnd_score*100:.1f}점
    - 종합 붕괴 위험도: {hype_index*100:.1f}%

    [AI 리스크 피처 해석 가이드라인 (반드시 준수)]
    1. R&D 리스크 점수 (높을수록 위험): 매출액 대비 R&D 비율이 낮거나(R&D 투자 부족), 상장 경과년수가 짧거나, 기술특례상장 기업인 경우 리스크 점수가 높게 나옵니다.
       - 주의: R&D 비율 자체는 높을수록 R&D 투자를 적극적으로 잘 하고 있는 우수 신호(낮은 리스크)이며, 낮을수록 기술 공동화 우려가 큰 위험 신호(높은 리스크)입니다. R&D 비율이 높은데도 R&D 리스크 점수가 높은 경우는 짧은 상장 연수나 기술특례상장 요인 때문입니다. R&D 투자를 많이 하는 행위를 \"R&D 공동화\" 또는 \"부정적 요인\"으로 설명하는 인과관계 오류를 절대 범하지 마십시오.
    2. 재무 리스크 점수 (높을수록 위험): 자본잠식률이나 부채비율이 높을수록, 연속 영업손실 연수가 길수록 점수가 높게 나옵니다.
    3. HYPE 리스크 점수 (높을수록 위험): 주가 변동성, 거래량 급증 비율, PSR/PBR 등이 높을수록 점수가 높게 나옵니다.
    
    [핵심 리스크 임계값 기준 (Slide 21 가이드라인)]
    1. 매출액: 연간 매출액이 30억 미만인 기간이 지속되면 상장적격성 실질심사 대상(위험)
    2. 정정 공시 빈도: 3개년 누적 3회 이상인 경우 불성실공시법인 지정 및 불투명성 극대화(위험)
    3. 자금 조달: 3개년 유상증자 횟수가 2회 이상이거나 빈번한 CB/BW 메자닌 발행으로 연명하는 경우 주주가치 희석 및 오버행 리스크 극대화(위험)
    4. 대주주 지분율: 최대주주 지분율이 20% 미만인 경우 경영권이 극도로 취약하며 적대적 M&A 및 대주주 이탈 위험(주의/위험)
    
    [기업 공시 및 재무 원본 데이터 (Combined CSV)]
    ```csv
    {csv_raw_text}
    ```
    
    [답변 원칙]
    1. 반드시 위에 제공된 원본 데이터(Combined CSV) 및 분석 스코어를 철저하게 참조하여 팩트 기반으로 답변하세요.
    2. 근거 없는 낙관론(예: '향후 큰 성장이 기대됩니다')은 피하고, 철저하게 리스크 예방적 관점에서 보수적이고 냉정하게 분석하세요.
    3. 숫자를 인용할 때는 원본 데이터의 금액(유상증자 액수 등)이나 비율을 정확하게 기재하세요.
    4. 질문자가 이해하기 쉽도록 친절하고 신뢰감 있는 전문 분석가 톤앤매너(한국어)로 작성해 주세요.
    5. 한자 금지: 답변 내용 전체에 한자(예: 株式, 負債 등)를 절대로 사용하지 마십시오. 오직 명확하고 쉬운 한글로만 답변해 주십시오.
    6. 마크다운 가독성: Streamlit 렌더링에 적합하도록 마크다운 문법(볼드체 **, 목록 -, 숫자 번호 등)을 활용하여 구조화되고 보기 좋게 답변을 작성해 주십시오.
    """
    
                messages_payload = [
                    {"role": "system", "content": system_prompt}
                ]
                for msg in st.session_state.messages[-10:]:
                    messages_payload.append({"role": msg["role"], "content": msg["content"]})
                    
                with st.spinner("AI가 공시 및 재무 데이터를 분석하고 답변을 작성하고 있습니다..."):
                    response_text, err = call_llm_api(
                        provider=ai_provider,
                        api_key=active_api_key,
                        model=active_model,
                        messages=messages_payload,
                        temperature=0.3
                    )
                    
                if err:
                    st.error(f"AI API 호출 중 오류가 발생했습니다: {err}")
                else:
                    response_text = response_text.replace("quyết정", "결정").replace("quyết", "결정")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()

        # 피처 설명 검색 (Feature Search) 메인 최하단 이동
        st.markdown("---")
        st.subheader(" 피처 설명 검색 (Feature Description Search)")
        st.markdown("Anti-Hype 엔진에서 사용하는 각 재무 및 주가 피처의 정의와 세부 기준을 검색합니다.")
        
        user_query = st.text_input("피처에 대해 질문하세요 (예: PSR, EBITAT 등)", key="main_feature_search_query")
        if user_query:
            if not active_api_key:
                st.warning(f"⚠️ 피처 설명 검색을 활성화하려면 사이드바에서 **{ai_provider} API Key**를 입력해 주세요.")
            else:
                messages = [
                    {"role": "system", "content": f"다음은 피처 설명 문서입니다. 질문에 답변하세요.\n\n{feature_doc}"},
                    {"role": "user", "content": user_query}
                ]
                with st.spinner("피처 설명을 검색하는 중..."):
                    answer, err = call_llm_api(ai_provider, active_api_key, active_model, messages)
                if err:
                    st.error(err)
                else:
                    st.markdown(f"""
                    <div style="background-color: {'#1e293b' if is_dark else '#f1f5f9'}; border: 1px solid {'#334155' if is_dark else '#cbd5e1'}; border-radius: 8px; padding: 15px; margin-top: 10px; color: {'#f8fafc' if is_dark else '#0f172a'};">
                        <strong>검색 답변:</strong><br>{answer}
                    </div>
                    """, unsafe_allow_html=True)

with tab2:
    st.subheader(" 2D 시계열 궤적 시각화 (2022년 01월 ~ 2025년 12월)")
    st.markdown("기업별 위험 점수의 시계열 변화를 2차원 좌표평면 상의 궤적으로 조망합니다. (Y축: HYPE 리스크, X축: 100 - 재무 리스크, 버블 크기: R&D 리스크)")
    if os.path.exists("trajectory_map.html"):
        with open("trajectory_map.html", "r", encoding="utf-8") as f:
            html_content = f.read()
            
        try:
            target_int_code = int(clean_code)
        except Exception:
            target_int_code = "null"
            
        if target_int_code != "null":
            html_content = html_content.replace(
                "let selectedCompanyCode = null;",
                f"let selectedCompanyCode = {target_int_code};"
            )
            
            injection_js = f"""const historyToggle = document.getElementById("history-toggle");
    
    // Auto-select company from Streamlit input on load
    if (selectedCompanyCode) {{
        const match = companies.find(c => c.stock_code === selectedCompanyCode);
        if (match) {{
            searchInput.value = `${{match.company_name}} (${{match.stock_code}})`;
        }}
    }}"""
            html_content = html_content.replace(
                'const historyToggle = document.getElementById("history-toggle");',
                injection_js
            )
            
        st.components.v1.html(html_content, height=850, scrolling=True)
    else:
        st.error("trajectory_map.html 파일을 찾을 수 없습니다. 먼저 generate_html_plot.py를 실행하여 생성해주세요.")
