### I am a graduate and currently preparing for my graduation project. I am focusing on fault diagnosis and machine learning. This is my weekly report
# Week 1 
Runge-Kutta

# Week 2  
Last week, I learnt Runge-Kutta.This week, I used the Runge-Kutta rule to conduct control simulations for the CSTR system  
The following is the mathematical model of the CSTR system:  

$$
\frac{dc_a}{dt} = -k_{0}e^{\frac{-E_{0}}{RT}}c_A + \frac{Q_{f}c_{af} - Q_{f}c_A}{Ah}
$$

$$
\frac{dT}{dt} = \frac{k_{0}e^{\frac{-E_{0}}{RT}}c_A(-\Delta H)}{\rho C_p} + \frac{Q_{f}T_{f}-Q_{f}T}{Ah} + \frac{UA_{c}(T_{c}-T)}{\rho C_pAh}
$$

$$
\frac{dT_c}{dt} = \frac{Q_c(T_{cf} - T_c)}{V_c} + \frac{UA_c(T-T_c)}{\rho_c C_pcV_c}
$$

$$
\frac{dh}{dt} = \frac{Q_{f} - Q}{A}
$$
  
Some parameters involved and their normal values can be found in the following literature:  
**Pattern Matching in Historical Data. Michael**  
I have established a single closed-loop control system involving temperature for the above system, as shown in the following figure:  
![image](https://github.com/Chris-Zouchenyu/Undergraduate-Graduation-Project-Report-git/blob/main/Closed_loop_control_system.png)  
When the reactor temperature changes, such as the set value from 402.35 to 390, the system responds as follows:  
![image](https://github.com/Chris-Zouchenyu/Undergraduate-Graduation-Project-Report-git/blob/main/response_result.png)  
# Week 3  
This week, I simulated the CSTH system, which is a system very similar to CSTR.  
This system comes from: A continuous stirred tank heater simulation model with applications.Nina F. Thornhill.doi::10.1016/j.jprocont.2007.07.006  
This is the physical model of the CSTH system:  

$$
\frac{dV(x)}{dt} = f_{cw} + f_{hw} - f_{out}(x)
$$

$$
\frac{dH}{dt} = W_{st} + h_{hw}\rho_{hw}f_{hw} + h_{cw}\rho_{cw}f_{cw} - h_{out}\rho_{out}f_{out}(x)
$$

$$
f_{out}(x) = 10^{-4}(0.1013x\sqrt{55+x} + 0.0237)
$$

$$
h_{out} = \frac{H}{V\rho_{out}}
$$
  
I handwritten the digital PI algorithm to perform PI control on the CSTH system. Under different PI parameters, the response of the system to changes in reactor temperature is shown in the following figure:  
![image](https://github.com/Chris-Zouchenyu/Undergraduate-Graduation-Project-Report-git/blob/main/CSTH_PID_parameters.png)  
It can be seen that as the Kp parameter increases, the time it takes for the system to stabilize becomes shorter, but the cost is a significant increase in the system's oscillation.  
