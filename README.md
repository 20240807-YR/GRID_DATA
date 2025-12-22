# GRID_DATA

## 전체 프로젝트 구조 (Core-based Architecture)
| Core | 핵심 질문 | 검증 대상 |
| :--- | :--- | :--- |
| **Core 1** | 스마트그리드의 출발점은 무엇인가? | **AMI + 예측 구조** |
| **Core 2** | ESS는 제어를 어떻게 바꾸는가? | **EMS + ESS 규칙** |
| **Core 3** | 이 구조는 하나의 두뇌로 통합 가능한가? | **통합 ML 모델** |
| **Core 4** | 안정성은 어디서 만들어지는가? | **Grid-forming 제어 논리** |
| **Core 5** | 이 구조는 전력망 밖에서도 유효한가 | **의료 IoT 확장** |

---

## Core별 상세 수행 계획 및 실제 작업 정리

###  Core 1 — AMI 기반 부하 데이터 + 예측 (스마트그리드의 출발점 정의)
* 핵심 목표: AMI의 역할을 “실시간 계측 + 예측”이라는 데이터 구조로 검증한다.
* 수행 내용
    * AMI 부하 데이터 로드
    * 기상 데이터 결합
    * 부하 예측 ML 모델 학습
    * Actual vs Predicted Load 생성
* 검증 포인트
→ 예측은 스마트그리드의 옵션이 아니라 구조적 필수 요소임을 확인

### Core 2 — ESS 관점 EMS 시뮬레이션 (배터리 중심 제어 구조 비교)
* 핵심 목표: ESS 유무에 따라 EMS 제어 결과가 어떻게 달라지는지 구조적으로 비교한다.
* 수행 내용
    * ESS 제약(SOC, 출력 한계) 반영
    *  동일 규칙 기반 Reactive / Proactive EMS 비교
    *  시계열 결과 및 지표 CSV 생성
* 검증 포인트
→ 규칙이 아니라 입력 정보의 밀도가 제어 효과를 결정함

### Core 3 — 통합 ML 모델 (AMI + ESS + EMS 결합)
* 핵심 목표
부하 예측과 ESS 제어 결과를 단일 모델 입력 구조로 통합한다.
* 수행 내용
    * Core 1 예측 결과 결합
    * Core 2 제어 결과 결합
    * 통합 Feature 구성
    * 단일 ML 모델 학습 및 저장
* 검증 포인트
→ 스마트그리드의 “통합 EMS 두뇌”가 데이터로 구현 가능함을 확인

### Core 4 — Grid-forming 관점 제어 구조 검증 (안정성 메커니즘 분석)
* 핵심 목표 : Grid-forming을 장비 기술이 아닌 제어 구조의 문제로 재정의한다.
* 수행 내용
    * 피크 억제 효과 분석
    * 변화율(Ramp rate) 안정성 비교
    * Reactive / Proactive 구조 차이 설명
* 검증 포인트
→ 안정성은 장비 성능이 아니라 제어 논리의 설계 결과

### Core 5 — 의료 IoT 확장 (스마트그리드 구조의 타산업 적용)
* 핵심 목표 : 스마트그리드 구조가 의료 IoT에서도 작동 가능한지 검증한다.
* 수행 내용
    * AMI–의료 센서 구조 대응
    * ESS SOC–환자 reserve 개념 대응
    * EMS–의료 AI 의사결정 대응
    * MySQL 기반 통합 DB 구조 설계
* 검증 포인트
→ 스마트그리드는 에너지 기술이 아닌 산업 인프라 구조

---

## 파일 정리
### 📅 12월 18일: AMI 데이터 기반 전력 소비–날씨 분석 및 예측
* **전력 소비–기온 관계 분석** (12_18_기온전력소비.ipynb)
    * `amiwea.csv` 파일을 로드하고 `timestamp`를 `datetime`으로 파싱했습니다.
    * `timestamp`를 `index`로 설정했습니다.
    * 기온(temperature)과 전력 소비(consumption) 간 산점도를 생성했습니다.
    * 동일 시간대 기준으로 전력 소비와 기온을 `이중 y축 시계열`로 시각화했습니다.
 
* **런던 스마트미터(AMI) 원천 데이터 컬럼 정리** (12_18_런던지표컬럼정리.ipynb)
    * `런던계절별사용량.csv` 파일을 로드했습니다.
    * 컬럼명을 분석용으로 변경했습니다.
        * `LCLid` → `household_id`
        * `DateTime` → `timestamp`
        * `KWHhh` → `consumption`
     * 특정 가구(MAC000001) 데이터만 필터링한 후 불필요한 `Index`, `Tariff` 컬럼을 삭제 후 `timestamp` 컬럼을 `index`로 설정했습니다.
     * `런던날씨.csv` 파일을 로드해 시간 기준으로 결합했습니다.이후 결합된 데이터를 `data_csv/amiwea.csv`로 저장했습니다.

* **시간대별 평균 전력 소비 패턴 분석**  (12_18_시간대별평균소비.ipynb)
    * `amiwea.csv` 파일을 로드하고 `timestamp`를 `index`로 설정했습니다.
    * 시간대 분석을 위해 `hour` 컬럼을 생성했고 시간대별 평균 전력 소비를 계산하고 그래프로 시각화했습니다.
    * 요일 분석을 위해 `weekday` 컬럼을 생성했으며 요일·시간대별 평균 전력 소비 패턴을 시각화했습니다.

* **단기 전력 부하 예측 모델 학습** (12_18_예측ML.ipynb)
    *	`amiwea.csv` 파일을 로드했습니다.
    *	시간 파생 변수(`hour`, `dayofweek`)를 생성했습니다. 그리고 지연 변수(`lag_1`, `lag_24`)를 생성했습니다.
    *	결측치를 제거해 학습용 데이터셋을 구성했으며 입력 변수와 타깃 변수를 정의했습니다.
    *	RandomForest 회귀 모델을 학습하여 예측 결과에 대해 MAE, RMSE를 계산했습니다.
    *	실제값과 예측값을 시계열로 시각화하여 `Feature importance`를 계산했습니다.예측 결과를 `pred_rf_test.csv`로 저장했습니다.
    *	XGBoost 회귀 모델을 학습하여 예측 결과를 `pred_xgb_test.csv`로 저장했습니다.

* **전력 소비 시계열 구조 확인** (12_18_전력소비시계열.ipynb)
    * `amiwea.csv` 파일을 로드하고 `timestamp`를 `index`로 설정했습니다.
    * `info()`를 통해 데이터 구조를 확인 후 시간 단위 전력 소비 시계열을 시각화했습니다.
    * 일 단위 평균 전력 소비로 resampling 후 시각화했습니다.
 
* **예측 기반 Grid-forming 안정성 지표 분석** (12_18_그리드포밍맥스.ipynb)
   * `XGB테스트.csv` 파일을 로드하고 `timestamp`를 `datetime`으로 파싱했습니다. 이후 `timestamp`를 `index`로 설정한 뒤 시간 순으로 정렬했습니다.
   * 실제 부하(actual) 기준 상위 5% 분위수를 계산해 `threshold`를 정의했습니다.
   * 운영 시나리오 구성
        * 예측이 없는 경우를 가정해 `load_naive`를 `actual`로 설정했습니다.
        * 예측이 있는 경우를 가정해, `pred_xgb`이 임계치를 초과할 경우 실제 부하를 10% 완화한 `load_predictive`를 생성했습니다.
   * 부하 변화량(Ramp) 계산
        * 각 시나리오별 부하 변화량의 절댓값을 계산했습니다.
             * ramp_naive
             * ramp_predictive
        * 시나리오별 최대 `ramp` 값을 계산했습니다.
        * 최대 부하 변화량을 주파수 불안정성의 대리 지표로 사용했습니다.
   * 극단적 부하 변화 구간 시각화
        * 초기 200개 시점에 대해 `naive` / `predictive` 시나리오의 `ramp`를 시계열로 비교했습니다.
        * 최대 `ramp` 발생 구간의 차이를 시각적으로 확인했습니다.
   * 고위험 Ramp 기준 재정의
        * `naive` / `predictive` 시나리오의 고위험 ramp 발생 횟수를 `bar chart`로 비교했습니다.
        * 고위험 `ramp` 구간을 임계선과 함께 시계열로 비교했습니다.
   * 결과 요약 테이블 생성
        * 시나리오별 최대 ramp 및 고위험 ramp 발생 횟수를 테이블로 정리했습니다.
         
* **실제 부하 및 예측 부하 데이터 로드**  (12_18_그리드포밍시뮬레이션.ipynb)
   * `amiwea.csv` 파일을 로드하고 `timestamp`를 `index`로 설정 후 `XGB테스트.csv` 파일을 로드하고 `timestamp`를 `index`로 설정했습니다.
   * 예측 결과가 포함된 데이터프레임을 운영 시뮬레이션용으로 복사했습니다.
   * 운영 시뮬레이션 공통 규칙 정의
        * 실제 부하(actual) 기준 상위 5% 분위수를 피크 임계치로 정의했습니다.
        * 예측 기반 완화 강도로 `alpha = 0.10`을 설정했습니다.
   * 운영 시나리오 정의
        * 예측이 없는 경우(`load_naive`)를 실제 부하로 설정했습니다.
        * 예측 기반 운영의 경우,예측값이 임계치를 초과할 것으로 판단되면 실제 부하를 10% 완화하도록 설정했습니다.
   * Peak Shaving 효과 계산
        * 각 시나리오에서 피크 임계치를 초과한 횟수를 계산했습니다.
        * `naive` / `predictive` 시나리오 간 피크 초과 횟수를 비교했습니다.
   * Ramp Smoothing 효과 계산
        * 각 시나리오에서 부하 변화량의 평균값을 계산했습니다.
        * 예측 기반 운영 시 평균 ramp 변화 여부를 확인했습니다.
   * 운영 시계열 비교
        * 초기 200개 시점에 대해 `naive` / `predictive` 운영 부하를 시계열로 비교했습니다.
        * 피크 임계선을 함께 표시했습니다.
   * 결과 요약 테이블 생성
        * 시나리오별 피크 초과 횟수와 평균 `ramp` 값을 테이블로 정리했습니다.

* **ESS 보조 운영 시나리오 구성** (12_18_ess.ipynb)
   * `XGB테스트.csv` 파일을 로드하고 `timestamp`를 `index`로 설정했습니다.
   * 실제 부하 기준 상위 5% 분위수를 피크 임계치로 정의했습니다.
   * 예측 기반 완화 비율(`alpha_pred = 0.10`)을 설정했습니다.
   * ESS 추가 완화 비율(`alpha_ess = 0.05`)을 설정했습니다.
   * 운영 시나리오 정의
        * 예측이 없는 경우(`load_naive`)를 실제 부하로 설정했습니다.
        * 예측 기반 운영(`load_predictive`)을 정의했습니다.
   * Peak Exceedance 비교
        * 각 시나리오에서 피크 임계치 초과 횟수를 계산했습니다.
        * `naive` / `predictive` / `predictive+ESS` 시나리오를 비교했습니다.
   * 부하 변화율(Ramp) 분석
        * 각 시나리오의 평균 부하 변화량을 계산했습니다.
        * ESS 보조 여부에 따른 변화 양상을 확인했습니다.
   * 고위험 Ramp 빈도 분석
        * `naive` 시나리오 기준 상위 5% ramp 값을 고위험 임계치로 정의했습니다.
        * 각 시나리오에서 고위험 ramp 발생 횟수를 계산했습니다.
   * 결과 요약 테이블 생성
        * 시나리오별 피크 초과 횟수, 평균 ramp, 고위험 ramp 횟수를 테이블로 정리했습니다.
   * 운영 시계열 비교 시각화
        * 초기 200개 시점에 대해 세 시나리오의 부하 시계열을 비교했습니다.
        * 피크 임계선을 함께 표시했습니다.

* **예측 기반 운영 + ESS 최적화 실험 구성** (12_18_데이터처리ess.ipynb)
   * `XGB테스트.csv` 파일을 로드하고 `timestamp`를 `datetime`으로 파싱 후 `timestamp`를 `index`로 설정한 뒤 시간 순으로 정렬했습니다.
   * 데이터가 정상 로드됐는지 `head()`로 확인했습니다.
   * `Baseline`을 **예측 기반 운영(Predictive, ESS 없음)**으로 정의했습니다.
   * 목적함수 J는 피크 초과, 위험 ramp, 평균 ramp를 baseline 대비 비율로 정규화해 가중합으로 구성했습니다.
   * ESS 사용량에 대한 페널티 항을 추가해 ESS 남용을 억제하도록 설정했습니다.
   * 최적화 변수는 `q_peak`, `alpha_pred`, `beta_ess`로 설정했습니다.
   * ESS 출력/에너지 용량은 가정 파라미터(상수)로 고정했습니다.
   * Baseline(Predictive, ESS 없음) 지표 계산
        * `baseline` 피크 임계치 분위수를 q_base = 0.95로 설정했습니다.
        * `actual` 기준 `threshold_base`를 계산했고 `baseline` 완화율을 `alpha_base = 0.10`으로 설정했습니다.
        * 예측값(pred_xgb)이 임계치를 넘는 경우에만 `actual`을 10% 완화한 `load_base`를 생성하여 `baseline` 지표를 계산했습니다.
   * 피크 초과 횟수 `peak_base`
        * ramp 시계열 `ramp_base = diff().abs()`
        * 위험 ramp 횟수 `risky_ramp_base` (ramp 상위 5% 초과)
        * 평균 ramp `avg_ramp_base`
        * `baseline` 결과를 출력해 수치가 생성됐는지 확인했습니다.
   * ESS 가정 파라미터 설정
        * ESS 최대 출력 `ESS_power_max = 5.0(kW)`, ESS 최대 에너지 `ESS_energy_max = 20.0(kWh)`로 설정했습니다.
        * 초기 SOC를 `SOC_init = 10.0(kWh)`로 설정했습니다.
   * 목적함수 가중치 설정
        * 피크 초과 항 가중치 `w1 = 0.45`로 설정했습니다.
        * 위험 ramp 항 가중치 `w2 = 0.45`로 설정했습니다.
        * 평균 ramp 항 가중치 `w3 = 0.10`로 설정했습니다.
        * ESS 사용량 페널티 항 가중치 `w4 = 0.10`로 설정했습니다.
   * Grid Search 탐색 범위 정의
        * 피크 임계치 분위수 후보를 `q_peaks = [0.90, 0.93, 0.95, 0.97, 0.99]`로 설정했습니다.
        * 예측 기반 완화율 후보를 `alpha_preds = np.arange(0.00, 0.32, 0.02)`로 설정했습니다.
        * ESS 보조 완화율 후보를 `beta_ess_list = np.arange(0.00, 0.32, 0.02)`로 설정했습니다.
   * 운영 시뮬레이션 및 목적함수 계산 함수 구현
        * `simulate_and_score(q_peak, alpha_pred, beta_ess)` 함수를 정의했습니다.
        * 입력 `q_peak`로부터 `actual` 분위수 기반 `threshold`를 계산했습니다.
        * SOC를 `SOC_init`으로 초기화했습니다.
        * 각 `timestamp`에 대해 다음 로직으로 부하를 계산했습니다.
        * 예측값이 임계치를 넘으면 `alpha_pred * actual`만큼 선제 완화(reduction) 적용했습니다.
        * 예측값이 임계치를 넘고 SOC가 남아있으면 ESS가 추가로 개입하도록 했습니다.
        * 요구량 `required = beta_ess * actual`을 계산했습니다.
        * ESS 출력은 `min(required, ESS_power_max, SOC)`로 제한했습니다.
        * ESS 출력만큼 SOC를 감소시켰고, SOC 하한을 0으로 제한했습니다.
        * 최종 부하는 `actual - reduction - ess_output`으로 기록했습니다.
        * ESS 사용량(ess_output)도 동시에 기록했습니다.
        * 시뮬레이션 결과로부터 지표를 계산했습니다.
             * 피크 초과 횟수 `Peak`
             * 위험 ramp 횟수 `RiskyRamp` (ramp 상위 5% 초과)
             * 평균 `ramp AvgRamp`
             * ESS 총 사용량 `ESS_energy`
        * 하드 제약 조건을 추가했습니다.
             * `(Peak > peak_base) and (RiskyRamp > risky_ramp_base)`이면 결과를 `None`했습니다.
        * 목적함수 J를 `baseline` 대비 비율로 정규화하여 계산했습니다.
             * `peak / peak_base`
             * `risky_ramp / risky_ramp_base`
             * `avg_ramp / avg_ramp_base`
             * `ess_energy / df["actual"].sum()`
   * `Grid Search` 실행 및 최적 결과 도출
        * 모든 (q_peak, alpha_pred, beta_ess) 조합에 대해 시뮬레이션을 수행했습니다.
        * 유효한 결과만 `results` 리스트에 저장했습니다.
        * `results_df` 데이터프레임으로 변환해 결과를 정리했습니다.
        * 목적함수 J 기준 오름차순 정렬 후 최적 조합(best)을 선택했습니다.
   * Baseline vs Optimized 비교
        * `Baseline(Predictive)`과 `Optimized(ESS + Params)` 시나리오를 비교하는 테이블(`comparison`)을 생성했습니다.
        * 비교 지표로 다음 값을 포함했습니다.
             * `Peak Exceedance`
             * `Risky Ramp`
             * `Average Ramp`
   * Trade-off 시각화
        * `Peak(x축)`과 `RiskyRamp(y축)`의 `trade-off` 산점도를 생성했습니다.
        * 점 색상은 목적함수 J 값으로 표현했습니다.
        * baseline 지점을 붉은 점으로 표시해 비교 기준을 고정했습니다.

* **ESS 제약 파라미터에 따른 trade-off 분석** (12_18_ess파라미터sweep&trade-off분석실패.ipynb)
   * `amiwea.csv` 파일을 로드하고 `timestamp`를 `index`로 설정 후 `XGB테스트.csv` 파일을 로드하고 `timestamp`를 `index`로 설정했습니다.
   * 예측값(pred_xgb)을 `df["pred"]`로 병합했습니다.
   * 공통 평가 지표 함수 정의
        * `evaluate_metrics(load_series, threshold)` 함수를 정의했습니다.
        * 부하 변화량 `ramp = diff().abs()`를 계산했습니다.
        * 지표를 다음 형태로 반환하도록 했습니다.
             * 피크 초과 횟수 `peak_exceed`
             * 평균 `ramp avg_ramp`
             * 위험 ramp 횟수 `risky_ramp` (상위 5% 초과)
   * ESS 제약 파라미터 공간 정의
        * ESS 출력 후보 `ESS_POWER_RANGE = [2, 4, 6, 8]`로 설정했습니다.
        * ESS 에너지 후보 `ESS_ENERGY_RANGE = [5, 10, 20]`로 설정했습니다.
        * 초기 SOC 비율을 `SOC_INIT = 0.5`로 설정했습니다.
   * ESS 제어 시뮬레이션 함수 구현
        * `simulate_ess(load, pred, power_max, energy_max, alpha=0.1)` 함수를 정의했습니다.
        * SOC를 `SOC_INIT * energy_max`로 초기화했습니다.
        * 예측값이 피크 임계치(`threshold`)를 초과하고 SOC가 남아 있으면 ESS 방전을 적용했습니다.
        * 방전량은 `min(power_max, alpha * pred, soc)`로 제한했습니다.
        * 부하에서 방전량을 차감하고 SOC를 감소시켰습니다.
   * 파라미터 Sweep 실행
        * 피크 임계치는 `consumption` 상위 5% 분위수로 정의했습니다.
        * 모든 (`power_max`, `energy_max`) 조합에 대해 ESS 시뮬레이션을 수행했습니다.
        * 시나리오별 지표를 계산해 `result_df`로 정리했습니다.
   * Trade-off 시각화
        * 에너지 용량별로 `power_max `대비 피크 초과 횟수를 선 그래프로 시각화했습니다.
        * `peak_exceed(x축)`와 `risky_ramp(y축)`의 `trade-off` 산점도를 생성했습니다.
        * 산점도 색상은 `power_max`로 표현했습니다.

⸻

### 📅 12월 19일: ESS 개입 규칙 변경 후 파라미터 Sweep
* **예측 기반 ESS 개입 규칙 변경 실험** (12_19_ess개입규칙변경.ipynb)
   * `amiwea.csv` 파일을 로드하고 `timestamp`를 `index`로 설정한 후 `XGB테스트.csv` 파일을 로드하고 `timestamp`를 `index`로 설정했습니다.
   * 예측값(pred_xgb)을 df["pred"]로 병합한 뒤 결측을 제거했습니다.
   * 공통 평가 지표 정의
        * `evaluate_metrics(load_series, peak_threshold, ramp_threshold)` 함수를 정의했습니다.
        * 부하 변화량 `ramp = diff().abs()`를 계산했습니다.
        * 지표로 다음 값을 반환했습니다.
             * `peak_exceed`
             * `avg_ramp`
             * `risky_ramp` (고정 ramp_threshold 초과)
   * Baseline 지표 산출
        * 피크 임계치를 `consumption` 상위 10% 기준(`quantile(0.90)`)으로 완화했습니다.
        * `baseline ramp` 기준은 consumption의 ramp 상위 5%로 설정했습니다.
        * baseline 지표를 계산해 `baseline_metrics`로 저장했습니다.
   * ESS 파라미터 공간 정의
        * `ESS_POWER_RANGE = [2, 4, 6, 8]`
        * `ESS_ENERGY_RANGE = [5, 10, 20]`
        * `SOC_INIT = 0.5`
        * `ALPHA = 0.1`
   * 규칙 변경된 ESS 제어 시뮬레이션 구현
        * `simulate_ess_predictive(load, pred, power_max, energy_max)` 함수를 정의했습니다.
        * 개입 조건을 다음과 같이 변경했습니다.
             * `pred > 0.9 * peak_threshold` 이고 `soc > 0`인 경우 개입
        * 방전량은 `min(power_max, ALPHA * pred, soc)`로 제한했습니다.
        * 부하에서 방전량을 차감하고 SOC를 감소시켰습니다.
   * 파라미터 Sweep 실행 및 결과 정리
        * `power_max`, `energy_max` 조합에 대해 부하 보정을 수행했습니다.
        * 각 시나리오별 지표를 계산해 `result_df`로 정리했습니다.
   * Trade-off 시각화
        * 에너지 용량별로 `power_max` 대비 피크 초과 횟수를 선 그래프로 시각화했습니다.
        * `peak_exceed`와 `risky_ramp`의 `trade-off` 산점도를 생성했고, 색상은 `power_max`로 표시했습니다.

* **pred 기준 평가 구조 재정의** (12_19_ess파라미터sweep&trade-off분석.ipynb)
   * `XGB테스트.csv` 파일을 로드하고 `timestamp`를 `index`로 설정 후 평가용 데이터프레임을 `pred_xgb → pred`로 컬럼명을 변경해 `df_eval`로 생성했습니다.
   * 데이터가 정상 생성됐는지 `head()`와 `shape`로 확인했습니다.
   * 공통 지표 함수 정의
        * `evaluate_metrics(load_series, threshold, ramp_threshold_fixed=None)` 함수를 정의했습니다.
        * ramp를 `diff().abs()`로 계산했습니다.
        * ramp 임계치가 입력되지 않으면 상위 5%로 자동 계산하도록 했습니다.
        * 반환값에 `ramp_threshold`도 포함하도록 구성했습니다.
   * Baseline에서 ramp 기준 고정
        * 피크 임계치는 actual 상위 5% 분위수로 정의했습니다.
        * baseline 부하는 actual 그대로 사용했습니다.
        * baseline 지표를 계산해 `ramp 임계치(ramp_threshold_fixed)`를 고정했습니다.
   * ESS 파라미터 정의 및 Sweep 실행
        * `ESS_POWER_RANGE = [2, 4, 6, 8]`
        * `ESS_ENERGY_RANGE = [5, 10, 20]`
        * `SOC_INIT = 0.5`
        * `ALPHA = 0.10`
        * 예측값이 임계치(threshold)를 초과하고 SOC가 남아 있으면 방전하도록 설정했습니다.
        * 방전량은 `min(power_max, alpha * actual, soc)`로 제한했습니다.
        * 모든 조합에 대해 결과를 계산하고 `result_df`로 정리했습니다.
   * Trade-off 시각화
        * 에너지 용량별로 `power_max` 대비 피크 초과 횟수를 선 그래프로 시각화했습니다.
        * `peak_exceed`와 `risky_ramp`의 `trade-off` 산점도를 생성했고, 색상은 `power_max`로 표시했습니다.

### 📅 12월 20일: Ramp 기반 ESS 개입 시뮬레이션
* **Ramp 기반 주파수 안정 proxy 정의 및 ESS 개입 실험** (12_20_ramp기반_ESS개입시뮬레이션.ipynb)
   * `XGB테스트.csv` 파일을 로드하고 `timestamp`를 `index`로 설정했고 `pred_xgb` 컬럼을 `pred`로 변경했습니다.
   * 데이터가 정상 생성됐는지 `head()`와 `shape`로 확인했습니다.
   * Ramp 기준 정의
        * actual의 ramp를 `diff().abs()`로 계산해 `df["ramp"]`로 생성했습니다.
        * baseline ramp 기준을 상위 5% 분위수(quantile(0.95))로 설정했습니다.
   * 공통 평가 지표 함수 정의
        * `evaluate_metrics(load_series, ramp_threshold)` 함수를 정의했습니다.
        * 평균 ramp(avg_ramp)와 위험 ramp 발생 횟수(risky_ramp)를 계산했습니다.
   * ESS 파라미터 공간 정의
        * `ESS_POWER_RANGE = [2, 4, 6, 8]`
        * `ESS_ENERGY_RANGE = [5, 10, 20]`
        * `SOC_INIT = 0.5`
   * Ramp 기반 ESS 개입 로직 구현
        * `simulate_ess_ramp_based(load, ramp, power_max, energy_max)` 함수를 정의했습니다.
        * 예측값은 사용하지 않고, ramp가 임계치를 초과하는 경우 즉각 개입하도록 설정했습니다.
        * 방전량은 `min(power_max, soc)`로 제한했습니다.
        * 부하에서 방전량을 차감하고 SOC를 감소시켰습니다.
   * Baseline 지표 계산
	        * baseline은 actual 그대로 두고 평균 ramp 및 위험 ramp 발생 횟수를 계산했습니다.
   * 파라미터 Sweep 실행 및 결과 정리
        * `power_max`, `energy_max` 조합에 대해 ramp 기반 ESS 개입 시뮬레이션을 수행했습니다.
        * 각 시나리오별 지표를 계산해 `result_df`로 정리했습니다.
        * baseline 포함 비교 테이블 summary를 생성했습니다.
   * 시각화 — ESS Power vs Risky Ramp
        * 에너지 용량별로 `power_max` 대비 위험 ramp 발생 횟수를 선 그래프로 시각화했습니다.
        * baseline 위험 ramp 값을 점선으로 함께 표시했습니다.

### 📅 12월 21일: 배터리·재생에너지·ESS 통합 데이터 처리 및 시뮬레이션

* **배터리 관련 원천 데이터 확인** (12_21_배터리.ipynb)
   * `changedis.csv` 파일을 로드 후 `info()`를 통해 데이터 타입과 결측 여부, 컬럼 목록을 확인했습니다. 이후 상위 3개 행을 출력해 데이터 구조를 확인했습니다.
   * `ev_battery.csv` 파일을 로드 후 `info()` 및 `columns`를 통해 배터리 스펙 관련 컬럼 구성을 확인했습니다. 이후 상위 3개 행을 출력해 데이터 구조를 확인했습니다.

* **재생에너지 예측 기반 ESS 시뮬레이션** (12_21_재생에너지예측ESS.ipynb)
   * `태양광재생에너지.csv` 파일을 로드 후 `DATE_TIME` 컬럼을 `datetime`으로 파싱하고 결측을 제거한 뒤 시간 순으로 정렬했습니다.
   * `AC_POWER` 기준으로 동일 시점의 발전량을 합산하고 컬럼명을 `generation_ac_kw`로 변경했습니다.
   * 시간 단위(1H)로 resample하여 평균 발전량을 계산하였으며 야간 및 결측 구간의 발전량을 0으로 보정했습니다.
   * 태양광 발전 예측 생성
        * 직전 시점 값을 사용하는 persistence 예측을 생성하였고 시간대(hour-of-day) 평균 발전량 패턴을 계산했습니다.
        * 두 예측을 가중합(`w = 0.7`)으로 결합해 예측 발전량을 생성한 후 음수 값은 0으로 clip 처리했습니다.
        * 예측 결과를 `generation_pred_kw` 컬럼으로 저장했습니다.
   * 정제된 재생에너지 CSV 생성
        * 컬럼명을 `timestamp`로 통일하고 `renewable_clean.csv` 파일로 저장했습니다.
   * AMI 부하 데이터 병합
        * `amiwea.csv` 파일을 로드하고 시간 기준으로 정렬하고 부하와 재생에너지 데이터를 `inner join`으로 병합했습니다.
        * 실측 발전량 기준 `net load`를 `net_load_no_forecast`로 생성하고 예측 발전량 기준 `net load`를 `net_load_with_forecast`로 생성했습니다.
   * 공통 EMS + ESS 운영 규칙 정의
        * `net load` 기준 상위 5%를 피크 임계치로 설정하고 예측 기반 완화율(`alpha_pred = 0.10`)을 설정했습니다.
        * ESS 추가 완화율(`alpha_ess = 0.05`)을 설정하였으며 ESS 출력·에너지·초기 SOC 파라미터를 정의했습니다.
   * EMS + ESS 시뮬레이터 구현
        * 예측 신호가 임계치를 초과할 경우 선제 완화를 적용한 후 ESS SOC가 남아 있을 경우 추가 방전을 적용했습니다.
        * 시나리오별 결과 부하 시계열을 반환하도록 구성했습니다.
   * 시나리오 실행
        * 예측 없음(reactive) 시나리오를 실행하고 태양광 예측 포함(proactive) 시나리오를 실행했습니다.
   * 지표 계산 및 결과 저장
        * 피크 초과 횟수, 평균 ramp, 위험 ramp 횟수를 계산했고 결과를 `core4_solar_ess_compare.csv`로 저장했습니다.
        * 시계열 비교 그래프를 생성했습니다.
   * 시간축 불일치 처리
        * 부하와 태양광 데이터의 시간 해상도가 다름을 확인한 후 시간 단위를 1시간으로 통일한 뒤 재결합했습니다.
        * Core 4에서는 재생에너지–ESS 단독 구조로 실험을 재구성했습니다.

* **태양광 예측 ESS 비교 실험 (에러 및 재구성)** (12_21_태양광예측ESS비교error.ipynb)
   * 태양광 원천 CSV의 컬럼 구조를 확인하고 태양광 발전량을 시간 단위로 resample했습니다.
   * persistence + 시간대 평균 기반 예측 발전량을 재생성하고 `renewable_clean.csv` 파일로 저장했습니다.
   * AMI 부하와 시간축 교집합 기준으로 병합한 후 net load를 기준으로 ESS 개입 시나리오를 구성했습니다.
   * 예측 없음(사후) / 예측 있음(선제) ESS 시나리오를 비교하고 결과를 `core4_solar_ess_compare.csv`로 저장했습니다.
   * 시계열 비교 그래프를 생성했습니다.

* **재생에너지·AMI·ESS 통합 시뮬레이션** (12_21_통합시뮬레이션.ipynb)
   * AMI, 풍력, 기존 ESS 시뮬레이션 결과 CSV를 로드하고 시간 컬럼을 자동 탐지해 `timestamp`로 통일했습니다.
   * 2013–2014 기간으로 데이터 범위를 제한하고 모든 데이터를 1시간 단위로 resample했습니다.
   * 풍력 발전량 예측(`persistence + hour-of-day`)을 생성한 후 `AMI + 풍력 데이터`를 병합해 `net load`를 구성했습니다.
   * 기존 `reactive/proactive ESS` 결과를 병합했습니다.
   * 학습용 데이터 구성
        * `proactive ESS`의 `discharge`를 `teacher` 신호로 설정했고 `AMI·풍력·net load·lag·시간대 변수`를 `feature`로 구성했습니다.
        * 결측 제거 후 시계열 분할(`train/test`)을 수행했습니다.
   * 모델 학습
        * `HistGradientBoostingRegressor` 모델을 학습하고 test 구간 예측 결과로 `MAE`, `RMSE`를 계산했습니다.
   * 제어 성능 비교
        * `baseline reactive` / `baseline proactive` / `learned control` 시나리오를 비교했습니다.
        * 공통 ESS 제약 조건 하에서 net load 조정 결과를 비교하고 결과 지표와 시계열 결과를 CSV로 저장했습니다.

* **풍력 기반 ESS 시뮬레이션** (12_21_풍력에너지ESS1.ipynb)
   * AMI 부하 데이터와 풍력 발전 데이터를 로드하고 2013–2014 구간으로 제한했습니다.그리고 시간 단위로 발전량을 정렬하여 net load를 구성했습니다.
   * 풍력 발전 예측(`persistence + hour-of-day`)을 생성하고 `reactive` / `proactive ESS` 시나리오를 실행했습니다.
   * 지표 계산 및 시계열 비교 그래프를 생성했습니다.

* **기타 데이터 확인 및 전처리** 
   * 원본 데이터 범위 확인 (12_21_data.ipynb)
        * `data.csv`를 로드하고 `datetime` 파싱 후 2013–2014 데이터 범위를 확인했습니다.
   * ESS 데이터 기반 시뮬레이션 (12_21_ESS데이터기반시뮬레이션.ipynb)
        * `changedis.csv`, `ev_battery.csv`를 로드 후 ESS 단위로 시간 집계를 수행했습니다.
        * EV 배터리 스펙을 추출해 ESS 파라미터로 사용하고 공통 규칙 기반 reactive / proactive EMS 시뮬레이션을 실행했습니다.
        * 지표 계산 및 결과 CSV를 생성했습니다.
   * NASA 데이터 전처리 (12_21_NASA전처리.ipynb)
        * `NASA.csv` 파일을 로드했고 문자열 형태의 `start_time`을 `datetime`으로 변환하여 2013–2014 데이터만 필터링했습니다.
    
* **의료판 ESS 시뮬레이션 구성** (12_21_의료판ESS시뮬레이션.ipynb)
   * `페이스메이커.csv` 파일을 로드하고 데이터 크기와 컬럼 목록을 확인했습니다.
   * 전·후 안정화 지표 생성
        * 현재 심박(`current_hr`)과 기준 심박(`baseline_hr`)의 차이를 `hr_error_before`로 계산했습니다.
        * 다음 심박(`next_hr`)과 기준 심박의 차이를 `hr_error_after`로 계산하고 심박 변화량을 `hr_ramp`로 계산했습니다.
        * SpO₂ 변화량을 spo2_ramp로 계산하여 주요 안정화 지표와 위험 이벤트 컬럼의 기초 통계를 확인했습니다.
   * 위험 임계치 정의
        * SpO₂ 위험 기준을 `92`로, SpO₂ 경고 기준을 `93`으로 설정하고 부정맥 위험 경고 기준을 `0.80`으로 설정했습니다.
        * 가상 개입 강도로 SpO₂ 보정량과 HR 완화 계수를 정의했습니다.
   * 시나리오별 개입 플래그 생성
        * Naive(사후): `next_spo2`가 위험 기준 미만일 때 개입하도록 설정했습니다.
        * Predictive(선제): 부정맥 위험이 높거나 SpO₂가 경고 기준 미만일 때 개입하도록 설정했습니다.
        * Context-aware: `Predictive` 조건에 저혈압(`sbp < 90`) 조건을 추가했습니다.
        * 각 시나리오별 개입 비율을 확인했습니다.
   * 의료판 ESS 개입 적용
        * 개입 시 `next_spo2`를 일정량 증가시키는 것으로 가정하여 `next_hr`를 `current_hr` 방향으로 완화하도록 보정했습니다.
        * 개입 이후 HR 오차, HR ramp, SpO₂ ramp를 재계산했고 개입 이후에도 SpO₂가 위험 기준 미만인 경우를 위험 이벤트로 카운트했습니다.
   * 시나리오 실행
        * `Baseline(무제어)` 시나리오를 구성하고, `Predictive`, `Context-aware` 시나리오를 각각 적용했습니다.
   * 결과 요약 테이블 생성
        * 평균 HR 오차(전/후), HR ramp, SpO₂ 위험 이벤트 개수를 계산하고 시나리오별 요약 테이블을 생성했습니다.
   * 시각화
        * 시나리오별 SpO₂ 위험 이벤트 개수를 막대 그래프로 비교하고 HR 안정화 전·후 평균 오차를 선 그래프로 비교했습니다.

* **페이스메이커 데이터 단독 분석** (12_21_페이스메이커분석.ipynb)
   * `페이스메이커.csv` 파일을 로드 후 데이터 타입과 컬럼 구성을 확인했습니다.
   * 제어 전·후 심박 안정화 확인
        * `hr_error_before`, `hr_error_after`를 계산하고 제어 이후 HR 오차가 평균적으로 감소하지 않음을 확인했습니다.
        * 데이터가 단기 안정화가 아닌 장기 보상 최적화를 목표로 설계되었음을 확인했습니다.
   * action(제어 여부) 기준 효과 비교
        * action별 HR 오차 및 reward 평균을 비교하고 개입(`action ≠ 0`)이 항상 reward를 증가시키지 않음을 확인했습니다.
   * 부정맥 위험 구간별 분석
        * `arrhythmia_risk`를 기준으로 `low/mid/high` 구간으로 분할한 후 위험 구간별 HR 오차와 reward 평균을 비교했습니다.
        * 고위험 구간에서는 HR 안정화보다 위험 억제가 우선됨을 확인했습니다.
   * 심박 ramp 분석
        * 심박 변화량(`hr_ramp`)을 계산하고 action별 평균 HR ramp를 비교했습니다. 이후 빠른 제어가 항상 동적 안정성을 보장하지 않음을 확인했습니다.
   * reward 분포 비교
        * action별 reward 분포 통계를 확인했고 진단명(diagnosis) 기준 reward 평균 비교 가능성을 확인했습니다.

* **Core5 최종 통합 실험** (12_21_Final.ipynb)
   * MySQL 데이터베이스(`core5_ems`)에 연결했고 Core4 EMS/ESS 데이터와 의료 IoT 데이터를 DB에서 로드했습니다.
   * 의료 IoT → EMS 구조 매핑
        * 심박을 전력 부하 상태로 대응시키고 부정맥 위험을 ESS SOC 개념으로 치환했습니다.
        * 심박 변화량을 trend 신호로 구성하였으며 EMS 입력에 필요한 최소 컬럼만 추출했습니다.
   * 공통 EMS 규칙 정의
        * 심박 기준 상위 5%를 임계치로 설정하고 개입 강도 `alpha = 0.10`을 적용했습니다.
        * Core4에서 학습된 통합 모델을 로드했습니다.
   * EMS 판단 입력 구성
        * `상태(state)`, `추세(trend)`, `여유(reserve)`로 입력 벡터를 구성했고 결측을 제거한 입력 행렬을 생성했습니다.
   * Reactive vs Proactive 개입 비교
        * 현재 상태 기준 reactive 신호를 구성하였으며 다음 상태 기준 proactive 신호를 구성했습니다.
        * 동일한 EMS 규칙으로 개입량을 계산했습니다.
   * 제어 결과 테이블 생성 및 저장
        * 제어 전·후 심박 시계열을 구성했고 결과를 DB 테이블(`core5_control_result`)로 저장했습니다.
        * 제어 결과 시계열을 시각화했습니다.

* **MySQL 기반 Core5 데이터 적재 및 통합** (12_21_mysql.ipynb)
   * MySQL 서버에 연결하고 core5_ems 데이터베이스를 생성하고 Core4 결과 CSV 및 AMI 데이터를 로드했습니다.
   * `timestamp` 기준으로 시간 축을 정렬하고 EMS–ESS 통합 테이블을 구성했습니다.
   * `의료 IoT placeholder` 컬럼을 추가한 후 e`ms_ess_core4` 테이블로 저장했습니다.
   * 페이스메이커 데이터 적재
        * `episode_id`와 `step`을 조합해 `timestamp`를 생성하고 시간 순으로 정렬했습니다.
        * `pacemaker_iot` 테이블로 저장했습니다.
   * Core4–IoT 통합 테이블 생성
        * 두 테이블의 시간 교집합 기준으로 병합 후 `core5_ems_ess_iot` 테이블로 저장했습니다. 이후 DB 내 테이블 목록과 row 수를 확인했습니다.
