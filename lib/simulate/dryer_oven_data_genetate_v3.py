import pandas as pd
import numpy as np
from datetime import datetime

"""
데이터 생성 조건
- 데이터는 건조기의 센서에서 생성하는 데이터 임.
- 컬럼은  timestamp / 내부 온도 / 습도 /  내용물의 무게 / 외부 온도 / 송풍기의 풍력 /  송풍기의 RPM / user_action
- 목표 내부 온도는 80도이다.
- 송풍기의 RPM에 따라서 풍력이 바뀌며 내부 온도에 영향을 끼친다
- 습도와 내용물의 무게 변화에 따라서 내부 온도가 바뀐다.
- 기본 입력으로 항시 20도의 열이 가해진다.
- 중간에 목표 내부 온도와 -40 ~ + 80의 변화가 4회 이상 발생한다.
- internal_temp가 target_temp에 미치지 못하면  다음 데이터에 rpm을 200 정도 낯추어서 전제 데이터를 갱신
- 반대로 높으면  rpm을 200 정도 높여서 전체데이터를 갱신
- 기본 샘플링이 완료 된 후에, 전체 데이터에서 0.2% 포지션에서 데이터 일정기간(duration) 동안 감소값(drop_value)만큼 감소했다가 다시 target_temp로 복구하는 코드를 넣어 주세요
- 상승도 동일한 원리 적용
- user_action은 현재 상태에 대한 rpm의 조치 값입니다. (0:down, 1:keep, 2:up)
"""
"""
1) 제시해주신 건조기 센서의 물리적 상관관계(RPM → 풍력 → 온도 영향, 습도/무게 → 온도 영향)와
입력 열기 20도 고정, 급격한 온도 변화(Target 80도 기준 -40~+80도) 조건을 반영한 파이썬 코드입니다.
2) 상시 상태(Steady State)에서 내부 온도가 목표 온도(80도) 대비 ±2도 이내에서 안정적으로 유지되도록 제어 로직을 수정
3) 사용자의 요구사항을 반영하여 내부 온도(internal_temp)가 목표 온도(target_temp)에 도달하지 못하면 다음 시점의 RPM을 낮추고, 높으면 RPM을 높여서 전체 시스템이 스스로 온도를 조절하는 피드백 제어 로직을 추가
"""

def generate_data_v3(drop_envs,
                     spike_envs,
                     csv_file_pathname,
                     num_samples=20000,
                     target_temp=80,
                     initial_temp=0,
                     with_ts=True,
                     base_input_heat=20.0,
                     verbose=0):
    """
    가상 데이터를 생성하여 csv 파일로 저장합니다.
    
    :param drop_envs: 하강 이벤트 정의
    :type drop_envs: set(ratio, duration, high)
    :param spike_envs: 하강 이벤트 정의
    :type spike_envs: set(ratio, duration, high)
    :param csv_file_pathname: csv 파일 이름
    :type csv_file_pathname: str
    :param num_samples: 전체 샘플의 수
    :type num_samples: int
    :param target_temp: 목표 온도
    :type target_temp: float
    :param initial_temp: 초기 온도
    :type initial_temp: float
    :param verbose: verbose
    :type verbose: int
    ;param base_input_heat: 항시 가해지는 기본 열
    ;type base_input_heat: int
    :return: csv_file_pathname
    :rtype: str
    """

    # 1. 초기 설정
    num_rows = num_samples
    base_heat = base_input_heat

    # drop 이벤트 인덱스 및 기간
    drop_env      = drop_envs[0]
    drop_idx      = int(num_rows * drop_env[0])
    drop_duration = drop_env[1]
    drop_value    = drop_env[2]

    # rise 이벤트 인덱스 및 기간
    spike_env     = spike_envs[0]
    rise_idx      = int(num_rows * spike_env[0])
    rise_duration = spike_env[1]
    rise_value    = spike_env[2]
    
    timestamps = np.arange(num_rows)
    now_timestamp = datetime.now().timestamp()
    for i in range(len(timestamps)):
        timestamps[i] = float(timestamps[i]) + now_timestamp
    
    # 데이터 저장 배열
    internal_temps, external_temps, pressures = np.zeros(num_rows), np.zeros(num_rows), np.zeros(num_rows)
    humidities, weights, rpms = np.zeros(num_rows), np.zeros(num_rows), np.zeros(num_rows)

    # 2. user_action 생성 (요청하신 조건 반영)
    # 0: DOWN, 1: KEEP, 2: UP
    user_actions = np.zeros(num_rows, dtype=int)
    user_actions[0] = 1  # 초기값은 1
    
    # 초기값
    curr_int_temp = initial_temp
    curr_ext_temp = base_heat
    curr_hum, curr_wgt = 50.0, 3.0
    curr_rpm = 1500.0  # 초기 RPM

    print(f'curr_int_temp = {curr_int_temp}')
    
    # 3. 시뮬레이션 루프
    for i in range(num_rows):
        is_drop = drop_idx <= i < (drop_idx + drop_duration)
        is_rise = rise_idx <= i < (rise_idx + rise_duration)

        # --- (1) RPM 피드백 제어 로직 (핵심 수정 사항) ---
        # 이전 단계의 온도를 확인하여 RPM 조절 (상시 구간에서 작동)
        if not (is_drop or is_rise):
            if curr_int_temp < target_temp - 0.5:
                curr_rpm -= 200.0  # 온도가 낮으면 RPM을 낮춤 (풍량 감소 -> 열 보존)
            elif curr_int_temp > target_temp + 0.5:
                curr_rpm += 200.0  # 온도가 높으면 RPM을 높임 (풍량 증가 -> 냉각)
            
            # RPM 작동 범위 제한 (현실적인 수치)
            curr_rpm = np.clip(curr_rpm, 1000, 3000)
        
        # --- (2) 이벤트 시 물리량 강제 변동 ---
        if is_drop:
            active_rpm = 500.0 + np.random.normal(0, 50)
            active_pressure = 0.5 + np.random.normal(0, 0.01)
        elif is_rise:
            active_rpm = 3500.0 + np.random.normal(0, 100)
            active_pressure = 1.8 + np.random.normal(0, 0.05)
        else:
            active_rpm = curr_rpm + np.random.normal(0, 30)
            active_pressure = 1.0 + (active_rpm / 2000)**2 * 0.2 + np.random.normal(0, 0.005)

        # --- (3) 외부 온도 및 건조 진행 ---
        weather = np.sin(i / (num_rows / 8)) * 1.5
        heat_leak = (curr_int_temp - curr_ext_temp) * 0.01
        
        if is_rise:
            curr_ext_temp = base_heat + rise_value + np.random.normal(0, 1.0)
        elif is_drop:
            curr_ext_temp = base_heat - drop_value + np.random.normal(0, 0.5)
        else:
            curr_ext_temp = base_heat + weather + heat_leak + np.random.normal(0, 0.1)

        curr_hum = max(10.0, curr_hum - np.random.uniform(0.01, 0.03))
        curr_wgt = max(1.0, curr_wgt - np.random.uniform(0.0005, 0.001))
        
        # --- (4) 내부 온도 계산 ---
        # RPM이 낮아지면(풍량이 줄면) 물리적으로 냉각 손실이 줄어 온도가 오르는 구조 반영
        physics_effect = (curr_ext_temp * 0.7) + (active_pressure * 20.0) - (active_rpm * 0.005) - (curr_hum * 0.1)
        
        # 기본 제어력 + 물리 효과
        next_temp = curr_int_temp + (target_temp - curr_int_temp) * 0.1 + (physics_effect * 0.01) + np.random.normal(0, 0.1)
        
        # --- (5) 이벤트 적용 및 상시 구간 안정화 ---
        if is_drop:
            next_temp = (target_temp - drop_value) + np.random.normal(0, 1.0)
        elif is_rise:
            next_temp = (target_temp + rise_value) + np.random.normal(0, 1.0)
        # else:
        #     # 피드백 덕분에 자연스럽게 유지되지만, 안전을 위해 범위를 좁게 clip
        #     next_temp = np.clip(next_temp, target_temp - 2.0, target_temp + 2.0)
        curr_int_temp = next_temp
        
        # 결과 저장
        internal_temps[i], external_temps[i], pressures[i] = round(curr_int_temp, 2), round(curr_ext_temp, 2), round(active_pressure, 3)
        humidities[i], weights[i], rpms[i] = round(curr_hum, 2), round(curr_wgt, 2), round(active_rpm, 0)

        # user_action 저장
        if abs(internal_temps[i] - target_temp) <= 2:
            user_actions[i] = 1
        else:
            if internal_temps[i] > target_temp:
                user_actions[i] = 2
            else:
                user_actions[i] = 0

        
    df = pd.DataFrame({
        'timestamp': timestamps,
        'inner_temp': internal_temps,
        'humidity': humidities,
        'weight': weights,
        'outer_temp': external_temps,
        'pressure': pressures,
        'fan_rpm': rpms,
        'user_action': user_actions
    })

    df.to_csv(csv_file_pathname, index=False)

    print("건조기 센서 가상 데이터 생성 완료!")
    print(f"데이터 샘플 (목표 온도: {target_temp}도):")
    # print(df.iloc[98:105]) # 변화 발생 지점 확인
    
    return csv_file_pathname