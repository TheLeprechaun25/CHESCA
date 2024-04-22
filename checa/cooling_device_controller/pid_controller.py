import numpy as np


class PIDController:
    def __init__(self):
        self.Kp = None
        self.Kd = None
        self.Ki = None
        self.dt = None

        self.Kp_outage = None
        self.OutTempScaler = None

        self.integral = 0.0  # integral of error for Ki
        self.prev_error = 0.0  # error last time step for Kd

        self.updated_integral = 0.0
        self.updated_prev_error = 0.0

    def get_actions(self, b, cur_value, setpoint_value, saturated_perc, outage_flag, outdoor_temp):
        """ Returns the actions for the current time step """
        error = setpoint_value - cur_value

        if saturated_perc < 0.04: # Update:
            self.updated_integral = self.integral
            # self.updated_integral = min(max(self.updated_integral, self.integrator_min), self.integrator_max)
            self.updated_prev_error = self.prev_error
        else: # there is saturation, reduce the integral and derivative terms proportionally
            self.integral = self.integral * (1.0 - saturated_perc)
            self.prev_error = self.prev_error * (1.0 - saturated_perc)

            self.updated_integral = self.integral
            # self.updated_integral = min(max(self.updated_integral, self.integrator_min), self.integrator_max)
            self.updated_prev_error = self.prev_error

        if not outage_flag:  # Normal operation
            self.integral = self.updated_integral + self.Ki * error * self.dt
            D = self.Kd * (error - self.updated_prev_error) / self.dt
            P = self.Kp * error
            self.prev_error = error

            pid_output = P + self.integral + D
        else:  # During outage
            P = self.Kp_outage * error
            pid_output = P + self.OutTempScaler * (outdoor_temp - cur_value)


        #if b == 1:
        #    print(f"new error: {error:.3f}, prev_error: {prev_error:.3f}, P: {P:.3f}, I: {integral:.3f}, D: {D:.3f}, Output: {pid_output:.3f}")

        return pid_output


