import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from models import model_BiLSTM, model_LSTM, model_MLP, model_Transformer
seed = 24
torch.manual_seed(seed)

SEQ_LEN = 100        # 输入长度：用过去10秒数据
PRED_LEN = 50       # 预测长度：预测未来5秒趋势
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flag = 'train'
INPUT_DIM = 4
HIDDEN_DIM = 64
def init_model(model_class):
    """初始化不同模型的实例（统一接口）"""
    if model_class == model_BiLSTM:
        return model_BiLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN).to(DEVICE)
    elif model_class == model_LSTM:
        return model_LSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN).to(DEVICE)
    elif model_class == model_Transformer:
        return model_Transformer(4, 64, 4, 2, SEQ_LEN, PRED_LEN).to(DEVICE)
    elif model_class == model_MLP:
        return model_MLP(input_dim=INPUT_DIM, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, pred_len=PRED_LEN).to(DEVICE)
    else:
        raise ValueError(f"不支持的模型类型：{model_class}")

def create_multistep_dataset(file_path, seq_len, pred_len):
    df = pd.read_excel(file_path) 
    # 取前三列 T, V, Wst
    raw_data = df.values[:, :3] 
    # 增加一阶差分特征 ---
    # 计算温度的变化率 (dT = T_current - T_prev)
    temp_diff = np.diff(raw_data[:, 0], axis=0).reshape(-1, 1)
    temp_diff = np.vstack(([0], temp_diff)) # 补齐第一行
    # 合并特征：现在有 4 个特征 [T, V, Wst, dT]
    data_extended = np.hstack((raw_data, temp_diff))

    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_extended)
    
    X, y = [], []
    for i in range(len(data_norm) - seq_len - pred_len + 1):
        X.append(data_norm[i : i + seq_len, :]) # 输入 4 个特征
        y.append(data_norm[i + seq_len : i + seq_len + pred_len, 0]) # 预测 T
    
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)), scaler

def plot_prediction_at_step(target_step, scaler, file_idx, model, model_name, save_path):
    # 提取温度 T 在归一化时的最小值和缩放比例
    # scaler.data_min_[0] 是温度的最小值，scaler.scale_[0] 是 (max-min) 的倒数
    t_min = scaler.data_min_[0]
    t_range = scaler.data_max_[0] - scaler.data_min_[0]
    with torch.no_grad():
        input_data = X[target_step].unsqueeze(0).to(DEVICE)
        model.load_state_dict(torch.load(save_path,map_location=DEVICE,weights_only=True))
        model.to(DEVICE)
        model.eval()
        pred_norm = model(input_data).cpu().numpy().flatten()
        true_norm = y[target_step].numpy()
        history_norm = X[target_step, -100:, 0].numpy()

        # 反归一化公式: Real = Norm * Range + Min
        pred_real = pred_norm * t_range + t_min
        true_real = true_norm * t_range + t_min
        history_real = history_norm * t_range + t_min
        pred_real = pred_real + (true_real[0] - pred_real[0])
        # 绘图
        plt.figure(figsize=(10, 5))
        plt.plot(range(-100, 0), history_real, 'k-', label='History (Real T)')
        plt.plot(range(PRED_LEN), true_real, 'b-', label='Actual Future (Real T)')
        plt.plot(range(PRED_LEN), pred_real, 'r--', label = f'{model_name} Prediction (Real T)')
        
        plt.axvline(x=0, color='gray', alpha=0.3)
        plt.title(f"Real Temperature Prediction at Step {target_step} | datadata_{file_idx+2}")
        plt.xlabel("Time Steps (relative to now)")
        plt.ylabel("Temperature (℃)")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(f'figure/data_{file_idx+2}_{model_name}_{target_step}', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭画布，释放内存

if __name__ == '__main__':
    filenames = [r'data\CSTH_2.xlsx',r'data\CSTH_3.xlsx',r'data\CSTH_4.xlsx']
    models = [model_BiLSTM, model_LSTM, model_Transformer, model_MLP]
    for file_idx,file in enumerate(filenames):
        # 加载数据
        X, y, scaler = create_multistep_dataset(file, SEQ_LEN, PRED_LEN)
        # 划分训练/测试 
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        for model_idx, model_cls in enumerate(models):
            model_name = model_cls.__name__  # 获取模型类名
            print(f"\n--- 训练模型：{model_name} ---")
            model = init_model(model_cls)# 实例化
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.L1Loss()
            # 训练
            if flag == 'train':
                print("开始训练多步预测模型{}...".format(model_name))
                model.train()
                for epoch in range(EPOCHS):
                    optimizer.zero_grad()
                    output = model(X_train.to(DEVICE))
                    loss = criterion(output, y_train.to(DEVICE))
                    loss.backward()
                    optimizer.step()
                    if (epoch+1) % 5 == 0:
                        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
                save_path = f"parameters/{model_name}_data_{file_idx+2}_epoch1000_lr{LEARNING_RATE}.pth" 
                torch.save(model.state_dict(), save_path)
                print(f"模型已保存到：{save_path}")
                plot_prediction_at_step(890,scaler,file_idx,model,model_name,save_path)
                plot_prediction_at_step(910,scaler,file_idx,model,model_name,save_path)
            else:
                save_path = f"parameters/model_{model_name}_data_{file_idx+2}_epoch1000_lr{LEARNING_RATE}.pth" 
                plot_prediction_at_step(900,scaler,file_idx,model,model_name,save_path)
                plot_prediction_at_step(910,scaler,file_idx,model,model_name,save_path)
    print('程序已经全部运行完毕')
    