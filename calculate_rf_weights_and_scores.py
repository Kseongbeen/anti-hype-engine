import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier

# Windows 터미널 한글 깨짐 방지
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    print("="*60)
    print(" Random Forest 기반 실체(Substance) 평가 가중치 산출 엔진")
    print("="*60)

    # 1. 데이터 파일 경로 설정
    flat_path = "csv/combined_financial_data_flat.csv"
    all_path = "csv/company_info_pharma_bio_all_filled.csv"

    if not os.path.exists(flat_path) or not os.path.exists(all_path):
        print(" 에러: 필요한 CSV 데이터 파일이 존재하지 않습니다.")
        print(f"   확인 필요: {flat_path}, {all_path}")
        return

    # 2. 데이터 불러오기 (인코딩 예외 처리)
    try:
        df_all = pd.read_csv(all_path, encoding='cp949')
    except Exception:
        df_all = pd.read_csv(all_path, encoding='utf-8')
        
    df_flat = pd.read_csv(flat_path, encoding='utf-8-sig')

    # 3. 데이터 정제 및 코드 매핑
    df_flat['clean_code'] = df_flat['기업코드'].astype(str).str.replace('=', '').str.replace('"', '').str.zfill(6)
    df_all['clean_code'] = df_all['stock_code'].astype(str).str.zfill(6)

    # 두 데이터 프레임 병합 (교집합)
    df = pd.merge(df_flat, df_all[['clean_code', 'drop_50_or_more_since_2025']], on='clean_code', how='inner')
    print(f"️ 분석 대상 기업 수 (재무 데이터 존재): {len(df)}개")

    # 4. 피처 엔지니어링 및 0~1 정규화 스케일링
    # (1) 매출액 점수 (30억 ~ 1000억 원 클리핑 후 Min-Max)
    rev_raw = df['2024_매출액(원)'].fillna(0)
    rev_clipped = np.clip(rev_raw, 3e9, 100e9)
    df['score_revenue'] = (rev_clipped - 3e9) / (100e9 - 3e9)

    # (2) R&D 비율 점수 (0% ~ 20% 클리핑 및 스케일링)
    rd_raw = df['2024_매출대비R&D비율(%)'].fillna(0)
    rd_clipped = np.clip(rd_raw, 0.0, 20.0)
    df['score_rd'] = rd_clipped / 20.0

    # (3) 자금조달 점수 (최근 3개년 유상증자 횟수 감점식)
    # 유상증자 조달액이 0보다 큰 경우 유상증자 실시로 판단
    u_2022 = df['2022_유상증자조달액(원)'].fillna(0) > 0
    u_2023 = df['2023_유상증자조달액(원)'].fillna(0) > 0
    u_2024 = df['2024_유상증자조달액(원)'].fillna(0) > 0
    paid_in_count = u_2022.astype(int) + u_2023.astype(int) + u_2024.astype(int)
    df['score_financing'] = 1.0 - (paid_in_count / 3.0)

    # (4) 자본잠식률 점수 (0% ~ 50% 클리핑 후 반전 스케일링: 낮을수록 1.0에 가까움)
    imp_raw = df['2024_자본잠식률(%)'].fillna(0)
    imp_clipped = np.clip(imp_raw, 0.0, 50.0)
    df['score_impairment'] = 1.0 - (imp_clipped / 50.0)

    # 5. ML 모델 학습용 데이터셋 구축
    feature_cols = ['score_revenue', 'score_rd', 'score_financing', 'score_impairment']
    feature_names = {
        'score_revenue': '1. 2024년 매출액 (가중치 50%)',
        'score_rd': '2. 2024년 R&D 비율 (가중치 25%)',
        'score_financing': '3. 3개년 유상증자 빈도 (가중치 17%)',
        'score_impairment': '4. 2024년 자본잠식률 (가중치 8%)'
    }
    
    X = df[feature_cols]
    y = df['drop_50_or_more_since_2025'].fillna(0)

    # 6. Random Forest Classifier 모델 정의 및 학습
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)

    # 모델 예측 점수 (훈련 정확도)
    train_accuracy = rf.score(X, y)
    
    # 피처 중요도 (가중치 비율)
    importances = rf.feature_importances_

    # 7. 결과 출력
    print("\n" + "-"*40)
    print(f" 1. 모델 설명력 (Train Accuracy): {train_accuracy * 100:.2f}%")
    print(f"   (2025년 이후 주가 50% 이상 폭락 여부 분류 모델 성능)")
    print("-"*40)
    
    print("\n 2. Random Forest 산출 피처 중요도 (Feature Importances):")
    for feat, imp in zip(feature_cols, importances):
        print(f"   - {feature_names[feat]}: {imp * 100:.2f}% (소수점 4자리: {imp:.4f})")
    print("-"*40)

    print("\n [요약 및 합의된 최종 가중치 스키마]")
    print("   Substance Score = (매출액 * 0.50) + (R&D비율 * 0.25) + (유상증자횟수 * 0.17) + (자본잠식률 * 0.08)")
    print("="*60)

if __name__ == "__main__":
    main()
