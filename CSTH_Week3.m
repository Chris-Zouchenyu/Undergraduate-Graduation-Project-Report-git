clc,clear
global hcw rolcw hhw rolhw u_current hout rolout

%物理参数
hcw = 100.6;%kj/kg
rolcw = 997.1;%kg/m3
hhw = 0;
rolhw = 0;
hout = @(t)4.1804*t + 0.2684;
rolout = @(t) -0.4134*t + 1007.6;

% 正常操作条件
x_nom = 20.48;%cm
fcw_nom = 9.038e-5;%m3/s
T_nom = 43.4;%℃
Wst_nom = 7.1844;%kj/s
H_nom = 57.7887;
fhw_nom = 0;
u_nom = [fcw_nom fhw_nom Wst_nom];
u_current = u_nom;

% 仿真参数（秒）
ts = 0.1;% 采样时间 (s)
t_total = 200;% 总仿真时间 (s)
tspan = 0:ts:t_total;% 时间轴
Nsim = length(tspan);% 仿真步数
fault_step = 190; % 故障引入时刻（对应19s）

V = 16*x_nom*(1e-6);%m-3
H = hout(T_nom)*rolout(T_nom)*V;
y0 = [V H];%初始条件

%% PI控制器参数
Kp = 0.05;% 比例系数
Ti = 1;% 积分时间常数
Ki = Kp*ts/Ti;% 积分系数
r = 30;% 温度设定值摄氏度

y_pi_control = zeros(2, Nsim);% PI控制后[V H]的状态
u_control_Wst = zeros(1, Nsim);% PI输出的Wst(kJ/s)
e_history = zeros(1, Nsim);% 误差历史
T_history = zeros(1,Nsim);% 温度历史
y_pi_control(:,1) = y0';
e_prev = 0;% 上一时刻误差
u_prev = Wst_nom;% 上一时刻控制量（初始为Wst稳态值）

for k = 2:Nsim
    % 提取上一时刻温度
    T_prev = H2T(y_pi_control(:, k-1));% 上一时刻温度
    T_history(k-1) = T_prev;% 存储温度
    e = r - T_prev;% 控制器的正反作用要搞明白
    e_history(k-1) = e;
    

    delta_u = Kp*(e - e_prev) + Ki*e;
    Wst = u_prev + delta_u;% PI输出直接作用于Wst
    
    % Wst限幅：参考文献Table1（0~15.04 kJ/s，蒸汽阀全关→全开）
    Wst = max(min(Wst, 15.04), 0);
    u_control_Wst(k-1) = Wst;
    

    u_current(3) = Wst ;
    u_current(1) = fcw_nom;% 固定fcw为稳态，避免液位耦合
    u_current(2) = fhw_nom;% 无热水进料
    
    % RK4求解
    h = ts;
    y_prev = y_pi_control(:, k-1);
    k1 = csth_model(y_prev);
    k2 = csth_model(y_prev + h * k1/2);
    k3 = csth_model(y_prev + h * k2/2);
    k4 = csth_model(y_prev + h * k3);
    y_now = y_prev + h * (k1 + 2*k2 + 2*k3 + k4) / 6;
    
    % 保存状态+更新PI历史值
    y_pi_control(:, k) = y_now;
    e_prev = e;
    u_prev = Wst;% PI上一时刻值为Wst
end
% 补全最后一个时刻的数值
u_control_Wst(end) = u_prev;
T_history(end) = H2T(y_pi_control(:, end));

%% 可视化
figure('Name','PI控制效果（Wst为主控变量）');
subplot(2,1,1);
plot(tspan, T_history, 'b-', 'LineWidth',1.2); hold on;
plot([0, t_total], [r, r], 'r--', 'LineWidth',1);
plot([fault_step*ts, fault_step*ts], [r-10, r+10], 'k:');
title('PI控制+RK4求解：加热器温度T（Wst主控）');
xlabel('时间(s)'); ylabel('温度(摄氏度)');
legend('实际温度','设定值','故障起始'); grid on;

subplot(2,1,2);
plot(tspan, u_control_Wst, 'g-', 'LineWidth',1.2);
title('PI控制器输出：蒸汽热输入Wst');
xlabel('时间(s)'); ylabel('Wst (kJ/s)');
grid on;
%% 模型函数
function dydt = csth_model(y)
    global hcw rolcw hhw rolhw rolout hout u_current
    V = y(1);H = y(2);
    fcw = u_current(1);fhw = u_current(2);Wst = u_current(3);
    x = V/(16e-4);
    T = H2T([V,H]);%单位摄氏度
    fout = @(x)1e-4*(0.1013*sqrt(55+x)+0.0237); 
    dV = fcw + fhw - fout(x);%单位m3/s
    dH = Wst + hcw*rolcw*fcw + hhw*rolhw*fhw - hout(T)*rolout(T)*fout(x);%单位KJ/s 焓平衡
    dydt = [dV;dH];
end

function x = H2T(y)
    x = NaN; 
    V = y(1);
    H = y(2);
    roots_num = Improved(-0.4134*4.1804,-0.4134*0.2684+4.1804*1007.6,1007.6*0.2684- H/V);
    for i = 1:length(roots_num)
        if abs(imag(roots_num(i))) < 1e-10
            real_root = real(roots_num(i));
            if real_root > 0 && real_root < 100
                x = real_root;
                x = round(x * 10000) / 10000; 
                break;
            end
        end
    end 
    if isnan(x)
        disp('提示：无0~100之间的实数根！');
    end
end

function roots=Improved(a, b, c)
    discriminant=b^2-4*a*c;
    sqrtDiscriminant=sqrt(discriminant);
    if b >= 0
        root1 = (-b-sqrtDiscriminant)/(2*a);
    else
        root1 = (-b + sqrtDiscriminant) / (2*a);
    end
    if root1~=0
        root2 = c / (a * root1);
    else
        root2 = -b / a;
    end
        roots = [root1, root2];
end