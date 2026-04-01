"""
gen_reward.py
- 보상(Reward) 생성 로직을 담당하는 모듈
- DryerEnv 클래스에서 호출되어 현재 온도와 행동에 따른 보상을 계산
- 다양한 버전의 보상 함수를 구현하여 실험 가능

"""
import math
import numpy as np

lib_name = "generate_reward"

   
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
    
def _generate_reward_v1(inner_temp, target_temp, action, temp_limit_depth=10.0):
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
            reward += 30.0 
        else: # UP or KEEP (팬 가속 -> 온도 추가 하락): 치명적 오류
            reward -= 30.0
    elif diff > 2.0: # [고온 상태] 80도 초과
        if action == 2:   # UP (팬 가속 -> 온도 하락 유도): 정답
            reward += 30.0
        else: # DOWN or KEEP (팬 감속 -> 온도 추가 상승): 치명적 오류
            reward -= 30.0

    # 3. 안정성 보상 (목표 도달 후 불필요한 떨림 방지)
    if abs(diff) <= 2.0 and action == 1: # 근접 상태에서 KEEP 유지
        reward += 15.0

    max_over_temp = temp_limit_depth + target_temp
    # 4. 과열 방지 패널티 (Safety)
    if inner_temp > max_over_temp:
        reward -= 300.0 # 특정 임계값 돌파 시 강력한 제재

    return reward


def _generate_reward_v0(inner_temp, target_temp, action, temp_limit_depth=10.0):
    """
    # Simple version
    # 보상(Reward)
    ; params inner_temp;
    ; return reward, terminated, truncated
    """
    diff = inner_temp - target_temp
    reward = -abs(diff)
    
    # 올바른 대응에 보너스
    if diff < -1.0 and action == 0: reward += 20.0 # 추울 때 RPM 낮추면 칭찬
    if diff > 1.0 and action == 2:  reward += 20.0 # 더울 때 RPM 높이면 칭찬
    # 목표 온도 근처(±2도)에 도달 시 추가 보상
    if abs(diff) <= 1.0:
        reward += 30.0

    # if diff > 1.0: # 온도가 높음
    #     reward += (20.0 if action == 2 else -30.0) # UP이면 상, DOWN/KEEP이면 큰 벌
    max_allow_temp = temp_limit_depth + target_temp
    min_allow_temp = temp_limit_depth
    if inner_temp > max_allow_temp or inner_temp < min_allow_temp:
        reward -= 100
    return reward


def _generate_reward_v2(inner_temp, target_temp, action, temp_limit_depth=10.0):
    """
    강제 탐색 유도형 보상 함수
    """
    # (온도 계산 로직: action 2 시 온도 하락, action 0 시 온도 상승 필수 반영)
    diff = inner_temp - target_temp
    reward = 0.0

    # [강제 탐색 로직] 상태와 액션의 부조화에 강력한 제재
    if diff > 1.0:  # 상황: 더움 (80도 초과) -> 반드시 UP(2)이 필요함
        if action == 2:    # 정답: UP
            reward += 30.0  # 보상을 다른 액션보다 훨씬 크게 설정
        elif action == 0:  # 오답: DOWN (온도를 더 높임)
            reward -= 50.0  # 매우 강력한 패널티로 이 선택을 기피하게 만듦
        else:              # KEEP
            reward -= 10.0
    elif diff < -1.0: # 상황: 추움 (80도 미만) -> 반드시 DOWN(0)이 필요함
        if action == 0:    # 정답: DOWN
            reward += 30.0
        elif action == 2:  # 오답: UP (온도를 더 낮춤)
            reward -= 50.0  # 2번 액션이 여기서 벌점을 먹으며 '추울 땐 안 하는 것'을 배움
        else:              # KEEP
            reward -= 10.0
    else:             # 상황: 적정 (79~81도)
        if action == 1:    # 정답: KEEP
            reward += 20.0
        else:              # 불필요한 조작
            reward -= 15.0

    # [중요] 타겟 도달 보너스 (거리 기반)
    reward -= abs(diff) * 5.0 

    return reward


def _generate_reward_v3(inner_temp, target_temp, action, temp_limit_depth=10.0):
    """
    정밀 수렴을 위한 보상 함수 (Exponential & Precision Reward)
    """
    diff = inner_temp - target_temp
    abs_diff = abs(diff)

    # 1. 지수적 보상 (80도에 가까울수록 보상이 폭발적으로 증가)
    # exp(-0.5 * abs_diff)는 0도 차이일 때 1.0, 10도 차이일 때 0.006을 반환함
    precision_reward = math.exp(-0.5 * abs_diff) * 50.0
    
    # 2. 거리 패널티 (멀어질수록 선형보다 강하게 패널티)
    dist_penalty = -(diff ** 2) * 0.5 

    # 3. 액션 가이드 (오차가 10도나 날 때 방치하지 않도록 강력 제재)
    action_guidance = 0
    if diff > 1.0: # 과열 상태
        action_guidance = 10.0 if action == 2 else -20.0 # UP(2) 유도
    elif diff < -1.0: # 부족 상태
        action_guidance = 10.0 if action == 0 else -20.0 # DOWN(0) 유도
    else: # 정밀 제어 구간 (±1도 이내)
        action_guidance = 15.0 if action == 1 else -10.0 # KEEP(1) 유도

    reward = precision_reward + dist_penalty + action_guidance

    # 4. 강제 탐색 보정 (10도 차이로 수렴하는 현상 방지)
    # 특정 오차(예: 5도 이상)가 지속되면 추가 패널티
    if abs_diff > 5.0:
        reward -= 50.0

    return reward

def _generate_reward_v4(inner_temp, target_temp, action, temp_limit_depth=10.0):
    """
    가우시안 보상
    """
    diff = inner_temp - target_temp
    
    # [수정 1] 가우시안 보상 (80도일 때 최대값 100, 멀어지면 급격히 0)
    # diff가 0일 때 exp(0)=1 -> 100점 / diff가 커질수록 exp(-값) -> 0점
    # 0.1은 종 모양의 너비(Sigma)입니다. 숫자가 작을수록 더 정밀하게 80도를 찾습니다.
    precision_reward = math.exp(-(diff**2) / (2 * (2.0**2))) * 100.0

    # [수정 2] 강력한 거리 패널티 (발산 방지 안전장치)
    # 온도가 target보다 높아질수록 마이너스가 무한히 커지게 설계
    dist_penalty = -abs(diff) * 5.0
    if inner_temp > target_temp:
        dist_penalty -= (diff**2) * 1.0 # 80도 초과 시 더 강한 패널티

    # [수정 3] 액션 가이드 (논리 강화)
    action_reward = 0
    if diff > 0.5:   # 80도 초과 (더움) -> UP(2) 필수
        action_reward = 10.0 if action == 2 else -50.0
    elif diff < -0.5: # 80도 미만 (추움) -> DOWN(0) 필수
        action_reward = 10.0 if action == 0 else -50.0
    else:             # 80도 근처 -> KEEP(1) 필수
        action_reward = 20.0 if action == 1 else -10.0

    reward = precision_reward + dist_penalty + action_reward
    return reward

def _generate_reward_v5(inner_temp, target_temp, action, temp_limit_depth=10.0):
    """
    가우시안 보상을 적용했음에도 70도에서 수렴한다는 것은, 모델이 "80도까지 가기 위해 
    팬을 더 낮추는 위험(DOWN)을 감수하는 것보다, 70도에서 적당한 패널티를 받으며 유지하는
    것이 누적 보상 측면에서 이득"이라고 판단한 국소 최적해(Local Optima)에 빠진 상태입니다.
    이 현상을 해결하려면 70도와 80도 사이의 '보상 절벽'을 없애고, 80도로 유도하는
    기울기(Gradient)를 더 가파르게 만들어야 합니다. 
    
    1. 정밀 수렴을 위한 "하이이브리드 보상" (Offset + Slope)
    가우시안의 좁은 보상 범위 때문에 70도 지점에서 보상을 찾지 못하는 문제를 해결하기 위해,
    거리에 비례한 선형 보상과 정밀 가우시안을 결합합니다.    
    """
    diff = inner_temp - target_temp  # (-)면 부족, (+)면 과열
    abs_diff = abs(diff)

    # [수정 1] 베이스 보상: 80도에 가까워질수록 '무조건' 커지게 (70도에서 80도로 끌어당기는 힘)
    base_reward = -abs_diff * 2.0 

    # [수정 2] 정밀 가우시안: 80도 근처(±2도)에서만 폭발적으로 보너스
    # Sigma를 1.0으로 좁혀서 80도에 딱 붙게 만듭니다.
    precision_bonus = math.exp(-(diff**2) / (2 * (1.0**2))) * 30.0

    # [수정 3] 방향성 강제 (70도에서 80도로 가려면 반드시 DOWN(0)이 필요함)
    direction_reward = 0
    if diff < -1.0:  # 79도 미만 (추움)
        if action == 0:   # DOWN: 온도를 올리는 유일한 방법
            direction_reward = 15.0 
        elif action == 2: # UP: 더 차갑게 만듦 (최악)
            direction_reward = -30.0
            
    elif diff > 1.0: # 81도 초과 (더움)
        if action == 2:   # UP: 온도를 낮추는 유일한 방법
            direction_reward = 15.0
        elif action == 0: # DOWN: 더 뜨겁게 만듦 (최악)
            direction_reward = -30.0

    reward = base_reward + precision_bonus + direction_reward

    # [수정 4] 정체 패널티: 70도 부근에서 계속 머물면 벌점 부여
    if 65.0 <= inner_temp <= 75.0:
        reward -= 5.0

    return reward


def _generate_reward_v6(inner_temp, target_temp, action, temp_limit_depth=10.0):
    """
    1. 폭주 방지형 "트랩(Trap) 보상"
    80도를 넘어서는 순간 보상을 절벽처럼 깎아버려야 모델이 무서워서 80도 위로 못 올라갑니다.
    """
    # [1] 물리 로직 (팬 UP시 확실히 온도가 깎여야 함)
    # heater(10) - cooling(1.0 * (temp-outer)) 구조인지 확인하세요!
    
    diff = inner_temp - target_temp 
    
    reward = 0.0

    # [2] 절대 구역 보상 (핵심: 80도 초과 시 강력한 응징)
    if diff > 0: # 80도 초과 (과열)
        reward -= (diff ** 2) * 5.0 # 제곱 패널티로 폭주 원천 차단
        if action == 2: reward += 10.0 # 그 와중에 UP(냉각)하면 칭찬
        else: reward -= 20.0           # DOWN이나 KEEP하면 엄벌
        
    elif diff < 0: # 80도 미만 (부족)
        reward -= abs(diff) * 1.5   # 80도로 유도하는 인력
        if action == 0: reward += 10.0 # DOWN(가열)하면 칭찬
        
    else: # 딱 80도!
        reward += 50.0 
        if action == 1: reward += 20.0 # KEEP 유지 시 보너스

    # [3] 액션 연속성 패널티 (지속 증가 방지)
    # 이전 액션과 반대되는 액션을 할 때 보상을 주어 진동 방지
    return reward


def _generate_reward_v7(inner_temp, target_temp, action, temp_limit_depth=10.0):
    """
    80도에서 '자석처럼' 멈추게 하는 최종 코드
    """
    # [2] 보상 함수: '80도'라는 구덩이에 빠뜨리기
    diff = inner_temp - target_temp
    
    # 핵심 1: 오차의 제곱에 마이너스를 붙임 (80도에서 멀어지면 무조건 감점)
    # 80도일 때 0점(최대), 100도나 60도일 때 -400점
    reward = -(diff ** 2) 

    # 핵심 2: 방향성 강제 보너스
    if diff > 2.0:   # 너무 뜨거우면(80도 초과)
        reward += (50.0 if action == 2 else -50.0) # UP하면 상, DOWN하면 벌
    elif diff < -2.0: # 너무 차가우면(80도 미만)
        reward += (50.0 if action == 0 else -50.0) # DOWN하면 상, UP하면 벌
    else:             # 80도 근처면
        reward += (50.0 if action == 1 else -10.0) # KEEP하면 상
    # [3] 안전 장치: 120도 넘어가면 에피소드 즉시 종료 및 대량 감점
    if inner_temp > 120.0 or inner_temp < 20.0:
        reward -= 500.0
    # # 핵심 2: 방향성 강제 보너스
    # if diff > 1.0:   # 너무 뜨거우면(80도 초과)
    #     reward += (20.0 if action == 2 else -50.0) # UP하면 상, DOWN하면 벌
    # elif diff < -1.0: # 너무 차가우면(80도 미만)
    #     reward += (20.0 if action == 0 else -50.0) # DOWN하면 상, UP하면 벌
    # else:             # 80도 근처면
    #     reward += (30.0 if action == 1 else -10.0) # KEEP하면 상
    # # [3] 안전 장치: 120도 넘어가면 에피소드 즉시 종료 및 대량 감점
    # if inner_temp > 120.0 or inner_temp < 20.0:
    #     reward -= 500.0
    return reward


def _generate_reward_v8(inner_temp, target_temp, action, temp_limit_depth=10.0):
    """
    80도에서 '자석처럼' 멈추게 하는 최종 코드
    """
    # [2] 보상 함수: '80도'라는 구덩이에 빠뜨리기
    diff = inner_temp - target_temp
    
    # 핵심 1: 오차의 제곱에 마이너스를 붙임 (80도에서 멀어지면 무조건 감점)
    # 80도일 때 0점(최대), 100도나 60도일 때 -400점
    reward = -(diff ** 2) 

    # 핵심 2: 방향성 강제 보너스
    if diff > 1.0:   # 너무 뜨거우면(80도 초과)
        # 선형이 아닌 제곱 패널티로 120도까지 가기 전에 모델을 겁줌
        reward = -(diff ** 2) * 2.0 
        # 과열 상태에서 DOWN(0)이나 KEEP(1)을 하면 가중 처벌
        if action != 2: reward -= 100.0
    else:             # 80도 근처면
        reward = -abs(diff) * 5.0
        if action == 0: reward += 10.0 # 추울 때 가열하면 보너스

    # [3] 안전 장치: 120도 넘어가면 에피소드 즉시 종료 및 대량 감점
    if inner_temp > 100.0:
        reward -= 1000.0
    return reward


def _generate_reward_v9(inner_temp, target_temp, action, temp_limit_depth=10.0):
    diff = inner_temp - target_temp
    abs_diff = abs(diff)
    
    # [A] 가우시안 정밀 보상: 80도에 딱 붙을수록 점수가 폭발 (최대 100점)
    # sigma를 0.5로 설정하여 매우 좁고 깊은 '보상 골짜기' 형성
    precision_reward = np.exp(-(diff**2) / (2 * (0.5**2))) * 100.0

    # [B] 오차 패널티 (P-제어)
    # 80도에서 멀어질수록 감점
    dist_penalty = -abs_diff * 5.0

    # [C] 액션 평탄화 패널티 (안정성 핵심)
    # 80도 근처(±1도)에서 KEEP(1)이 아닌 UP/DOWN을 하면 감점 (떨림 방지)
    stability_penalty = 0
    if abs_diff < 1.0 and action != 1:
        stability_penalty = -10.0
    elif abs_diff < 1.0 and action == 1:
        stability_penalty = 20.0 # 유지 성공 보너스

    # [D] 방향성 가이드 (과열/부족 판단)
    guidance = 0
    if diff > 1.0 and action == 2: guidance = 10.0 # 과열 시 UP
    if diff < -1.0 and action == 0: guidance = 10.0 # 부족 시 DOWN

    reward = precision_reward + dist_penalty + stability_penalty + guidance

    if inner_temp > 110.0 or inner_temp < 20.0:
        reward -= 500.0
    return reward


def generate_reward(inner_temp, action, target_temp, prev_inner_temp=None, temp_limit_depth=10.0):
    run_this = False
    gen_func=(_generate_reward_v0,
              _generate_reward_v1,
              _generate_reward_v2,
              _generate_reward_v3,
              _generate_reward_v4,
              _generate_reward_v5,
              _generate_reward_v6,
              _generate_reward_v7,
              _generate_reward_v8,
              _generate_reward_v9)
    reward = gen_func[8](inner_temp=inner_temp,
                         target_temp=target_temp,
                         action=action,
                         temp_limit_depth=temp_limit_depth)
    if run_this:
        reward += _generate_delta_reward(prev_inner_temp=prev_inner_temp,
                                         current_temp=inner_temp,
                                         target_temp=target_temp,
                                         action=action)
    return reward
