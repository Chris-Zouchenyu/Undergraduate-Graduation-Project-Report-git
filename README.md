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
Pattern Matching in Historical Data. Michael  
I have established a single closed-loop control system involving temperature for the above system, as shown in the following figure:  
  
When the reactor temperature changes, such as the set value from 402.35 to 390, the system responds as follows:  
