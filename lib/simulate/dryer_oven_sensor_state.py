"""
    DryerOvenSensorState 클래스는 건조기 오븐의 센서 상태를 나타내는 클래스입니다.
"""


class DryerOvenSensorState:
    """
       건조기 오븐의 센서 상태를 나타내는 클래스입니다.
       이 클래스는 건조기 오븐의 센서 데이터를 캡슐화하여,
       시뮬레이터에서 상태를 표현하는 데 사용됩니다.
    """
    def __init__(self, *args):
        """
        :param args: 센서 데이터 (timestamp, inner_temp, error, humidity, weight, outer_temp, pressure, fan_rpm)
        """
        items = args        
        if len(args) == 1:
            items = items[0] if isinstance(items[0], (list, tuple)) else items
        if len(items) != 8:
            raise ValueError("Expected 8 arguments")
        else:
            self.timestamp = args[0]
            self.inner_temp = args[1]
            self.error = args[2]
            self.humidity = args[3]
            self.weight = args[4]
            self.outer_temp = args[5]
            self.pressure = args[6]
            self.fan_rpm = args[7]

    # Make the object iterable
    def __iter__(self):
        """
            Allows unpacking the object like: timestamp, inner_temp,
            error, humidity, weight, outer_temp, pressure, fan_rpm = state
        """
        yield self.timestamp
        yield self.inner_temp
        yield self.error
        yield self.humidity
        yield self.weight
        yield self.outer_temp
        yield self.pressure
        yield self.fan_rpm

    def __str__(self):
        """
            String representation of the sensor state for easy debugging and visualization.
        """
        return f"Timestamp: {self.timestamp}, Inner Temp: {self.inner_temp:.2f}, Error: {self.error:.2f}, Humidity: {self.humidity:.2f}, Weight: {self.weight:.2f}, Outer Temp: {self.outer_temp:.2f}, Pressure: {self.pressure:.2f}, Fan RPM: {self.fan_rpm:.1f}"
