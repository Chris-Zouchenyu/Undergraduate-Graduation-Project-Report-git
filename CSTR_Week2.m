% 数字式PI控制
% CSTR仿真 采用单闭环控制系统 冷却剂的流量Qc作为控制变量 调控反应器的温度
clc; clear; close all;
%% 全局参数
global A rolcp k0 Vc E0R u_current UAc
% 物理参数
% 注意单位
A = 0.1666;% 反应器横截面积 (m²)
rolcp = 239;% 反应液热容 (J/(L·K))
k0 = 7.2e10;% 反应速率常数前项 (1/min)
Vc = 10;% 冷却剂体积 (L)
E0R = 8750;% 活化能/气体常数 (K)
UAc = 5e4;% 总传热系数 (J/(min·K))

% 正常操作条件
Qf_nom = 100;% 进料流量 (L/min)
Tf_nom = 320;% 进料温度 (K)
Caf_nom = 1;% 进料浓度 (mol/L)
Qc_nom = 15;% 冷却剂流量 (L/min)
Tcf_nom = 300;% 冷却剂温度 (K)
Q_nom = 100;% 反应器出口物料流量(L/min)
u_nom = [Qf_nom, Tf_nom, Caf_nom, Qc_nom, Tcf_nom, Q_nom];
u_current = u_nom;  % 全局操作变量

% 仿真参数（秒）
ts = 0.1;% 采样时间 (s)
t_total = 500;% 总仿真时间 (s)
tspan = 0:ts:t_total;% 时间轴
Nsim = length(tspan);% 仿真步数
fault_step = 190; % 故障引入时刻（对应19s）
x0 = [0.037, 402.35, 345.44, 0.6];  % 初始状态 [Ca, T, Tc, h]
%% PI控制器参数
% 采用单闭环控制系统 冷却剂的流量Qc作为控制变量 调控反应器的温度
Kp = 5;% 比例系数
Ki = 0.01;% 积分系数
r = 390;% 温度设定值K


x_pi_control = zeros(4, Nsim);% PI控制后的状态
u_control_Qc = zeros(1, Nsim);% PI输出的Qc
e_history = zeros(1, Nsim);% 误差历史
x_pi_control(:,1) = x0';
e_prev = 0;% 上一时刻误差
u_prev = Qc_nom;% 上一时刻控制量
%% PI控制+故障工况仿真
for k = 2:Nsim
    % PI控制逻辑
    T_prev = x_pi_control(2, k-1);% 上一时刻温度
    e = T_prev - r;% 误差 符号很重要！
    e_history(k-1) = e;
    
    % 增量式PI计算
    delta_u = Kp*(e - e_prev) + Ki*e;
    u_Qc = u_prev + delta_u;
    
    % 控制量限幅
    u_Qc = max(min(u_Qc, 30), 5);
    u_control_Qc(k-1) = u_Qc;
    if k == 2
        u_control_Qc(k-1) = u_nom(4);
    end
    % 更新全局操作变量（Qc为PI输出）
    u_current(4) = u_Qc ;
    
    % RK4求解
    h = ts;
    x_prev = x_pi_control(:, k-1);
    k1 = cstr_model(x_prev);
    k2 = cstr_model(x_prev + h * k1/2);
    k3 = cstr_model(x_prev + h * k2/2);
    k4 = cstr_model(x_prev + h * k3);
    x_now = x_prev + h * (k1 + 2*k2 + 2*k3 + k4) / 6;
    
    % 保存状态+更新PI历史值
    x_pi_control(:, k) = x_now;
    e_prev = e;
    u_prev = u_Qc;
end
% 补全最后一个控制量
u_control_Qc(end) = u_prev;
%% 可视化(AI写的)
% PI控制效果
figure('Name','PI控制效果');
subplot(2,1,1);
plot(tspan, x_pi_control(2,:), 'b-', 'LineWidth',1.2); hold on;
plot([0, t_total], [r, r], 'r--', 'LineWidth',1);
plot([fault_step*ts, fault_step*ts], [r-30, r+30], 'k:');
title('PI控制+RK4求解：反应器温度T');
xlabel('时间(s)'); ylabel('温度(K)');
legend('实际温度','设定值','故障起始'); grid on;

subplot(2,1,2);
plot(tspan, u_control_Qc, 'g-', 'LineWidth',1.2);
title('PI控制器输出：冷却剂流量Qc');
xlabel('时间(s)'); ylabel('Qc (L/min)');
grid on;
%% 函数
function dxdt = cstr_model(x)
    %'''
    % 输入 x = [Ca T Tc h]
    % 输出 dxdt 微分方程
    %'''
    global A rolcp k0 Vc E0R u_current UAc
    
    % 解包状态和操作变量
    Ca = x(1); T = x(2); Tc = x(3); h = x(4);
    Qf = u_current(1); Tf = u_current(2); Caf = u_current(3);
    Qc = u_current(4); Tcf = u_current(5);Q = u_current(6);
    
    % 反应速率常数 (1/min)
    k = k0 * exp((-E0R)/T);
    
    % 微分方程（min）
    dCa = -k*Ca + (Qf*Caf - Qf*Ca)/(1000*A*h);  % mol/(L·min)
    dT = k*Ca*UAc/rolcp + (Qf*Tf - Qf*T)/(1000*A*h) + UAc*(Tc-T)/(1000*rolcp*A*h);  % K/min
    dTc = Qc*(Tcf-Tc)/Vc + UAc*(T-Tc)/(4175*Vc);  % K/min
    dh = (Qf - Q)/A;  % m/min
    
    % 秒
    dxdt = [dCa; dT; dTc; dh]/60;
end