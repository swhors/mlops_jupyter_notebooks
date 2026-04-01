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
    def __init__(self, use_ts, target_temp=80.0, initial_temp=25.0, initial_weight=80, dry_weight=20, debug=False):
        """
        Initializes the dryer oven simulation.

        :param target_temp: The target temperature for the dryer oven.
        :param initial_temp: The initial temperature of the dryer oven.
        """
        self.target_temp = target_temp
        self.current_temp = initial_temp
        self.time_step = 0
        self._debug = debug
        self._dry_weight = dry_weight
        self._initial_weight = initial_weight
        self._use_ts = use_ts

    def _calculate_humidity_and_weight(self, inner_temp, outer_temp, humidity, weight):
         # 3. [추가] 수분 증발 및 무게 감소 로직
        # 온도가 높고 수분이 많을수록 증발이 잘 일어남
        evap_k = 0.00005 
        evaporation = evap_k * humidity * (inner_temp - outer_temp)
    
        # 무게 감소 반영 (단위: kg)
        weight -= evaporation
        weight = max(self._dry_weight, weight) # 건조물 자체 무게 이하로는 안 빠짐
    
        # 수분 함유율 업데이트
        total_water = self._initial_weight - self._dry_weight
        current_water = weight - self._dry_weight
        humidity = max(0.0, current_water / total_water)
        return humidity, weight

    def _calculate_inner_temp(self, humidity, inner_temp, outer_temp, fan_rpm, weight, heater_input=15.0):
        q_heater = heater_input
        # 2. [추가] RPM에 따른 내부 압력 계산 (Fan Law)
        # 외압 1013hPa 기준, RPM 제곱에 비례하여 압력 상승
        pressure_k = 0.000005
        inner_pressure = 1013.0 + pressure_k * (fan_rpm**2)

        humidity, weight = self._calculate_humidity_and_weight(inner_temp=inner_temp,
                                                               outer_temp=outer_temp,
                                                               humidity=humidity,
                                                               weight=weight)
            
        cooling_coeff = (inner_pressure / 1013.0) * 0.25
        q_cooling = cooling_coeff * (inner_temp - outer_temp)
        
        # 3. 수분 증발 손실 (수분이 많을수록 온도가 안 오름)
        # self.moisture: 0.0 ~ 1.0 (수분 함유율)
        q_evaporation = humidity * (inner_temp * 0.04) 

        # 4. 최종 온도 변화
        inner_temp += (heater_input - q_cooling - q_evaporation)
        if self._debug:
            print(f'heater_input={heater_input}, q_cooling={q_cooling}, q_evaporation={q_evaporation}')
        return inner_temp, humidity, weight
    
    def step(self, action, state, heater_input=15.0):
        """
        Simulates one time step in the dryer oven environment based on the given action and current state.

        :param action: The action taken by the agent (0: DOWN, 1: KEEP, 2: UP)
        :param state: The current sensor state of the dryer oven
        ;param heater_input: 히터는 일정한 열을 공급
        :return: A new sensor state after applying the action
        """
        if self._debug:
            print(f'lib.simulate.dryer_oven_simulation.DryerOvenSimulation.step.action={action}')
        if self._use_ts:
            timestamp, inner_temp, error, humidity, weight, outer_temp, pressure, fan_rpm = state
        else:
            inner_temp, error, humidity, weight, outer_temp, pressure, fan_rpm = state
        # 습도 및 무게 자연 감소
        # humidity = max(15.0, humidity - 0.01)
        # weight = max(4.2, weight - 0.0001)

        # 1. 액션에 따른 팬 RPM 변화
        if action == 0:   # DOWN: 팬 속도 감소 -> 냉각 효율 저하 -> 온도 상승 유도
            fan_rpm -= 200
        elif action == 2: # UP: 팬 속도 증가 -> 냉각 효율 상승 -> 온도 하락 유도
            fan_rpm += 200
        
        fan_rpm = np.clip(fan_rpm, 200, 4000) # 가동 범위
        
        new_inner_temp, humidity, weight = self._calculate_inner_temp(humidity=humidity,
                                                                      inner_temp=inner_temp,
                                                                      outer_temp=outer_temp,
                                                                      fan_rpm=fan_rpm,
                                                                      weight=weight,
                                                                      heater_input=heater_input)
        if self._debug:
            print(f"Action: {action}, Fan RPM: {fan_rpm:.1f}, Cooling Factor: {cooling_factor:.3f}, Heat Loss: {heat_loss:.2f}, Temp Change: {temp_change:.2f}, New Inner Temp: {new_inner_temp:.2f}, Pre Inner Temp: {inner_temp:.2f}" )
        if self._use_ts:
            return {"timestamp": timestamp,
                    "inner_temp": new_inner_temp,
                    "error": new_inner_temp - self.target_temp,
                    "humidity": humidity,
                    "weight": weight,
                    "outer_temp": outer_temp,
                    "pressure": pressure,
                    "fan_rpm": fan_rpm}
        return {"inner_temp": new_inner_temp,
                "error": new_inner_temp - self.target_temp,
                "humidity": humidity,
                "weight": weight,
                "outer_temp": outer_temp,
                "pressure": pressure,
                "fan_rpm": fan_rpm}
