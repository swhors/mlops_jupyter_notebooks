import pandas as pd
import numpy as np
from datetime import datetime


def generate_data_v1(drop_envs,
                     spike_envs,
                     csv_file_pathname,
                     num_samples=20000,
                     target_temp=80,
                     initial_temp=0,
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
    :return: None
    :rtype: None
    """
    print(f'Start to generate sample data. [{datetime.now()}]')

    # 시간축 (초 단위 가정)
    time = np.arange(num_samples)
    now_timestamp = datetime.now().timestamp()
    for i in range(len(time)):
        time[i] = float(time[i]) + now_timestamp

    # 2. 내부 온도(inner_temp) 생성 로직
    inner_temp = np.full(num_samples, target_temp, dtype=float)

    # 초기 가열 구간 (0~10%)
    heating_end = int(num_samples * 0.1)
    inner_temp[:heating_end] = np.linspace(initial_temp, target_temp, heating_end)

    # 급격한 온도 하강 시나리오 (예: 문 열림)
    if verbose > 0:
        print(f'급격한 온도 하강 시나리오 (예: 문 열림)')
    for drop_env in drop_envs:
        ratio = drop_env[0]
        duration = drop_env[1]
        drop_high = drop_env[2]
        drop_start = int(num_samples * ratio)
        if verbose > 0:
            print(f'급격한 온도 하강 시나리오 생성[{drop_start}, {duration}, {drop_high}]')
        inner_temp[drop_start:drop_start+duration] -= np.linspace(0, 30, duration)
        # 복구 구간
        inner_temp[drop_start+duration:drop_start+duration*2] = np.linspace(inner_temp[drop_start+duration-1], target_temp, duration)

    if verbose > 0:
        print('# 급격한 온도 상승 시나리오 생성 (예: 히터 과열)')
    for spike_env in spike_envs:
        ratio = spike_env[0]
        duration = spike_env[1]
        spike_high = spike_env[2]
        spike_start = int(num_samples * ratio)
        if verbose > 0:
            print(f'급격한 온도 상승 시나리오 생성[{spike_start}, {duration}, {spike_high}]')
        inner_temp[spike_start:spike_start+duration] += np.linspace(0, spike_high, duration)
        # 복구 구간
        inner_temp[spike_start+duration:spike_start+duration*2] = np.linspace(inner_temp[spike_start+duration-1], target_temp, duration)

    if verbose > 0:
        print('# 미세 노이즈 추가 (실제 데이터 느낌 부여)')
    inner_temp += np.random.normal(0, 0.4, num_samples)

    if verbose > 0:
        print('# 기타 센서 데이터 생성')
    humidity = np.linspace(80, 15, num_samples) + np.random.normal(0, 1, num_samples) # 습도 점진적 감소
    weight = np.linspace(5.0, 4.2, num_samples) + np.random.normal(0, 0.01, num_samples) # 무게 감소
    outer_temp = np.random.normal(22, 1, num_samples) # 외부 온도 (평균 22도)
    pressure = np.random.normal(1013, 2, num_samples) # 대기압

    # 4. 사용자 조치 및 팬 RPM 결정 (Down:0, Keep: 1, Up:2)
    def determine_action(temp):
        if temp < 78: return "0"   # 온도가 낮으면 팬 속도 감소(열 보존)
        elif temp > 82: return "2" # 온도가 높으면 팬 속도 증가(열 배출)
        else: return "1"           # 적정 온도 유지

    actions = [determine_action(t) for t in inner_temp]

    # Action에 따른 팬 RPM 수치 할당
    fan_rpm = []
    for action in actions:
        if action == "0": fan_rpm.append(1200 + np.random.randint(-50, 50))
        elif action == "2": fan_rpm.append(1800 + np.random.randint(-50, 50))
        else: fan_rpm.append(1500 + np.random.randint(-30, 30))

    # 5. 데이터프레임 생성 및 확인
    df = pd.DataFrame({
        'timestamp': time,
        'inner_temp': np.round(inner_temp, 2),
        'humidity': np.round(humidity, 2),
        'weight': np.round(weight, 3),
        'outer_temp': np.round(outer_temp, 2),
        'pressure': np.round(pressure, 2),
        'fan_rpm': fan_rpm,
        'user_action': actions
    })

    # 결과 출력 (상위 10개 및 하강 구간 확인)
    if verbose > 0:
        print("--- 초기 가열 구간 데이터 ---")
        print(df.head(10))
        print("\n--- 온도 하강 발생 구간 데이터 ---")
        for drop_env in drop_envs:
            ratio = drop_env[0]
            duration = drop_env[1]
            print(df.iloc[drop_start:drop_start + duration])
        print("\n--- 온도 상승 발생 구간 데이터 ---")
        for spike_env in spike_envs:
            ratio = spike_env[0]
            duration = spike_env[1]
            # 급격한 온도 상승 시나리오 (70% 지점, 예: 히터 과열)
            spike_start = int(num_samples * ratio)
            print(f"\n--- 급격한 온도 상승 구간 데이터 (ration = {ratio}) ---")
            print(df.iloc[spike_start:spike_start + duration])
    # CSV 파일로 저장
    df.to_csv(csv_file_pathname, index=False)
    print(f'End to generate sample data. [{datetime.now()}]')