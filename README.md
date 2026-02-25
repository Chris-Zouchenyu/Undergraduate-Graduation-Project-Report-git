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
<p align="center">
  <img src="https://github.com/Chris-Zouchenyu/Undergraduate-Graduation-Project-Report-git/blob/main/Closed_loop_control_system.png" width="600">
</p>
When the reactor temperature changes, such as the set value from 402.35 to 390, the system responds as follows:  
<p align="center">
  <img src="https://github.com/Chris-Zouchenyu/Undergraduate-Graduation-Project-Report-git/blob/main/response_result.png" width="600">
</p>
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
# Week 4  
This week, I used the CSTH system simulation model I built to simulate the fault conditions.I generated simulation data using MATLAB and then classified the fault conditions using LSTM.  

### LSTM model  

The calculation process of the Long Short Term Memory Network (LSTM) at time step $t$ is as follows:

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(C_t)
$$

**Symbol Description:**
- $f_t, i_t, o_t$: They are respectively the forget gate, input gate, and output gate.
- $C_t$: Unit state (long-term memory).
- $h_t$: Hidden state (short-term memory).
- $\odot$: Hadamard  
# Week 5-6  
This week, I used Transformer for real-time prediction of fault data. The structure of Transformer is shown in the following figure  
References: A novel transformer-based multi-variable multi-step prediction method for chemical process fault prognosis.  
doi: https://doi.org/10.1016/j.psep.2022.11.062  
<p align="center">
  <img src="https://www.cnblogs.com/nickchen121/p/16470765.html#gallery-1" width="600">
</p>
The forecast results are shown in the following figure:  
<img src="img/data_3_model_LSTM_900.png" width="200" />
<img src="img/data_3_model_MLP_900.png" width="200" />
<img src="img/data_3_model_BiLSTM_900.png" width="200" />
<img src="img/data_3_model_Transformer_900.png" width="200" />
The four images represent the results of using LSTM MLP BiLSTM Transformer to predict temperature in the case of data_3 (valve failure). The fault was introduced in step 890, and the temperature was predicted in step 900. The results show that LSTM and MLP perform poorly in response to temperature changes, with large errors and delays exceeding 1 second (step 10). The Transformer performs well with a delay of 1 second.  