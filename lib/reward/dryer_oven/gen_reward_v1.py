"""
gen_reward.py
- 보상(Reward) 생성 로직을 담당하는 모듈
- DryerEnv 클래스에서 호출되어 현재 온도와 행동에 따른 보상을 계산
- 다양한 버전의 보상 함수를 구현하여 실험 가능

"""


lib_name = "reward_generator"

   
def _generate_delta_reward(prev_inner_temp, current_temp, target_temp, action):
    """
    이전 온도와 현재 온도의 변화량을 이용하여 보상을 세밀하게 조정하는 함수
    """
    # 이전 값과의 관계 계산 (델타값)
    if prev_inner_temp is None:
        temp_delta = 0
    else:
        temp_delta = current_temp - prev_inner_temp # (+)면 상승 중, (-)면 하락 중
            
    reward = 0.0
    diff = current_temp - target_temp # 목표와의 거리

    # 2. 변화 관계를 이용한 정밀 보상 로직
        
    # [상황 A] 온도가 목표보다 낮을 때 (추운 상태)
    if diff < -0.5:
        if temp_delta > 0: # 온도가 상승 중이라면 (좋은 흐름)
            reward += 5.0
        else:              # 온도가 정체되거나 하락 중이라면 (나쁜 흐름)
            reward -= 5.0
         # 액션과의 관계: 추운데 온도가 내려가고 있는데 DOWN(0)을 했다면 보너스 (열 보존)
        if temp_delta <= 0 and action == 0:
            reward += 10.0
     # [상황 B] 온도가 목표보다 높을 때 (더운 상태)
    elif diff > 0.5:
        if temp_delta < 0: # 온도가 하락 중이라면 (좋은 흐름)
            reward += 5.0
        else:              # 온도가 정체되거나 상승 중이라면 (나쁜 흐름)
            reward -= 5.0
                
        # 액션과의 관계: 더운데 온도가 올라가고 있는데 UP(2)을 했다면 보너스 (냉각 강화)
        if temp_delta >= 0 and action == 2:
            reward += 10.0

    # 3. 목표 근처 안정성 (변화량 최소화 유도)
    if abs(diff) < 1.0:
        if abs(temp_delta) < 0.1: # 온도가 변하지 않고 안정적임
            reward += 15.0
        else:                     # 목표 근처에서 온도가 출렁임
            reward -= 5.0
        
    return reward
    
def _generate_reward_v2(inner_temp, target_temp, action, temp_limit_depth=10.0):
    diff = inner_temp - target_temp  # (+)면 과열, (-)면 부족
    reward = 0.0

    # 1. 정밀 구간별 기본 보상 (Dead-zone 설정)
    if abs(diff) <= 1.0:
        reward += 20.0  # 최적 구간 (매우 높은 보상)
    elif abs(diff) <= 2.0:
        reward += 5.0   # 허용 구간 (약간의 보상)
    else:
        reward -= abs(diff) * 2.0 # 이탈 구간 (거리에 비례한 패널티)

    # 2. 방향성 강제 보상 (가장 중요: 80도 위/아래를 명확히 분리)
    if diff < -2.0:  # [저온 상태] 80도 미만
        if action == 0:   # DOWN (팬 감속 -> 온도 상승 유도): 정답
            reward += 10.0 
        elif action == 2: # UP (팬 가속 -> 온도 추가 하락): 치명적 오류
            reward -= 30.0
        else:
            reward -= 30.0
    elif diff > 2.0: # [고온 상태] 80도 초과
        if action == 2:   # UP (팬 가속 -> 온도 하락 유도): 정답
            reward += 10.0
        elif action == 0: # DOWN (팬 감속 -> 온도 추가 상승): 치명적 오류
            reward -= 30.0
        else:
            reward -= 30.0

    # 3. 안정성 보상 (목표 도달 후 불필요한 떨림 방지)
    if abs(diff) <= 2.0 and action == 1: # 근접 상태에서 KEEP 유지
        reward += 15.0

    max_over_temp = temp_limit_depth + target_temp
    # 4. 과열 방지 패널티 (Safety)
    if inner_temp > max_over_temp:
        reward -= 300.0 # 특정 임계값 돌파 시 강력한 제재

    return reward

def generate_reward(inner_temp, action, target_temp, prev_inner_temp=None, temp_limit_depth=10.0):
    reward = _generate_reward_v2(inner_temp=inner_temp,
                                 target_temp=target_temp,
                                 action=action,
                                 temp_limit_depth=temp_limit_depth)
    reward += _generate_delta_reward(prev_inner_temp=prev_inner_temp,
                                     current_temp=inner_temp,
                                     target_temp=target_temp,
                                     action=action)
    return reward
