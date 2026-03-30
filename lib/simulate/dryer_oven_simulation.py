"""
DryerOvenSimulation 클래스는 건조기 오븐의 시뮬레이션을 제공합니다.
"""
import numpy as np
from lib.simulate.dryer_oven_sensor_state import DryerOvenSensorState


classname = "DryerOvenSimulation"


class DryerOvenSimulation:
    """
    DryerOvenSimulation 클래스는 건조기 오븐의 시뮬레이션을 제공합니다.
    """
    def __init__(self, target_temp=80.0, initial_temp=25.0):
        """
        Initializes the dryer oven simulation.

        :param target_temp: The target temperature for the dryer oven.
        :param initial_temp: The initial temperature of the dryer oven.
        """
        self.target_temp = target_temp
        self.current_temp = initial_temp
        self.time_step = 0
    
    def step(self, action, state):
        """
        Simulates one time step in the dryer oven environment based on the given action and current state.

        :param action: The action taken by the agent (0: DOWN, 1: KEEP, 2: UP)
        :param state: The current sensor state of the dryer oven
        :return: A new sensor state after applying the action
        """
        timestamp, inner_temp, error, humidity, weight, outer_temp, pressure, fan_rpm = state
        # 습도 및 무게 자연 감소
        humidity = max(15.0, humidity - 0.01)
        weight = max(4.2, weight - 0.0001)

        # 1. 액션에 따른 팬 RPM 변화
        if action == 0:   # DOWN: 팬 속도 감소 -> 냉각 효율 저하 -> 온도 상승 유도
            fan_rpm -= 100
        elif action == 2: # UP: 팬 속도 증가 -> 냉각 효율 상승 -> 온도 하락 유도
            fan_rpm += 100
        
        fan_rpm = np.clip(fan_rpm, 0, 3000) # 가동 범위
    
        # 2. 열 평형 방정식 (핵심 로직)
        # 히터는 일정한 열을 공급 (Heater Input)
        heater_input = 10.0 
    
        # [핵심] 팬 RPM에 비례하는 냉각 계수 (Cooling Coefficient)
        # RPM이 커질수록 cooling_factor가 커져서 빼지는 값이 커짐
        cooling_factor = fan_rpm / 1500.0  # 기준 RPM 1500일 때 1.0
    
        # 외부 온도와의 차이에 의한 열 손실 (Heat Loss)
        # 팬이 빠를수록(cooling_factor ↑), 내부가 더울수록(temp diff ↑) 손실이 커짐
        heat_loss = cooling_factor * (inner_temp - outer_temp) * 0.1

        # 3. 최종 온도 변화
        # 공급(Heater)보다 손실(Loss)이 크면 온도는 떨어짐 (UP 액션 시 발생)
        # 공급(Heater)보다 손실(Loss)이 작으면 온도는 올라감 (DOWN 액션 시 발생)
        temp_change = heater_input - heat_loss
        new_inner_temp = inner_temp + temp_change

        # 4. 결과 검증 (인지 확인)
        # - UP(2) 클릭 -> RPM 상승 -> cooling_factor 상승 -> heat_loss 상승 -> temp_change 감소 -> 온도 하락
        # - DOWN(0) 클릭 -> RPM 하락 -> cooling_factor 하락 -> heat_loss 하락 -> temp_change 증가 -> 온도 상승
        print(f"Action: {action}, Fan RPM: {fan_rpm:.1f}, Cooling Factor: {cooling_factor:.3f}, Heat Loss: {heat_loss:.2f}, Temp Change: {temp_change:.2f}, New Inner Temp: {new_inner_temp:.2f}" )
        return {
                "timestamp": timestamp,
                "inner_temp": new_inner_temp,
                "error": new_inner_temp - self.target_temp,
                "humidity": humidity,
                "weight": weight,
                "outer_temp": outer_temp,
                "pressure": pressure,
                "fan_rpm": fan_rpm}
