# CIFAR-10 图像分类神经网络实现

本项目实现了一个手工搭建的三层神经网络分类器，目标是通过 **CIFAR-10** 数据集进行图像分类。代码使用 **numpy** 完成模型的反向传播计算，涵盖了从数据预处理、神经网络模型构建到训练、测试以及超参数调优等功能，完全不依赖于任何深度学习框架（如 TensorFlow、PyTorch 等）。

## 项目结构

```
.
├── README.md          # 项目说明文档
├── main.py            # 主程序
├── requirements.txt   # 依赖库
└── best_model_weights.npz  # 最佳模型权重（训练完毕后保存）
```

## 环境要求

- `numpy`
- `matplotlib`
- `seaborn`
- `tensorflow`（仅用于加载CIFAR-10数据集）

可以通过以下命令安装所需的依赖：

```bash
pip install -r requirements.txt
```


## 代码说明

### 1. 数据加载与预处理

`load_and_preprocess_data()` 函数加载 **CIFAR-10** 数据集，并进行数据归一化、展平以及标签的 One-Hot 编码。

### 2. 神经网络模型

该神经网络由以下两层组成：
- 第一层：输入层到隐藏层
- 第二层：隐藏层到输出层

支持自定义激活函数：
- `relu(x)`：ReLU 激活函数
- `tanh(x)`：Tanh 激活函数
- `Sigmoid(x)`：Sigmoid 激活函数

模型通过反向传播计算梯度并更新权重。

### 3. 训练过程

训练过程中，我们实现了以下功能：
- **SGD优化器**：使用随机梯度下降法进行训练。
- **学习率衰减**：可选的学习率衰减策略，可以在每个epoch结束时逐渐减小学习率。
- **交叉熵损失**：通过交叉熵损失函数来优化模型。
- **L2正则化**：为了防止过拟合，模型实现了L2正则化。
- **模型保存**：根据验证集损失自动保存最优模型的权重。

### 4. 超参数调优

为了优化模型性能，我们对以下超参数进行了调优：
- **学习率**：从 0.001、0.01、0.1 中选择最优值。
- **隐藏层大小**：可以选择 64、128 或 256 个神经元作为隐藏层大小。
- **L2 正则化强度**：在 0.001、0.01 和 0.1 之间进行调整。

调优过程中，记录了不同超参数下的训练结果，并选择最优的超参数配置。

### 5. 测试过程

训练完成后，模型会保存最佳权重到 `best_model_weights.npz` 文件。如果你已经有训练好的模型，可以直接使用 `test_model()` 函数加载权重并进行测试。

### 6. 可视化

- 绘制训练集和验证集的损失曲线和准确率曲线。
- 可视化神经网络的权重和偏置。

## 代码运行

### 1. 训练与超参数调优

在代码中，使用 `hyperparameter_search()` 函数进行训练并选择最佳超参数。该函数会自动调整学习率、隐藏层大小和正则化强度，然后在 CIFAR-10 数据集上进行训练，输出最佳测试准确率。

```bash
python main.py
```

### 2. 测试模型

训练完成后，模型会保存最佳权重到 `best_model_weights.npz` 文件。如果你已经有训练好的模型，可以直接使用 `test_model()` 函数加载权重并进行测试。

```python
# 加载模型权重并进行测试
model = NeuralNetwork(input_size=3072, hidden_size=128, output_size=10, activation_fn=tanh, activation_derivative_fn=tanh_derivative)
model.load_weights('best_model_weights.npz')
test_model(model, X_test, Y_test)
```

### 3. 保存与加载模型

训练完成后，可以使用 `save_model_weights()` 函数将模型权重保存到 `.npz` 文件中，稍后可以重新加载并进行测试。

```python
# 保存模型权重
save_model_weights(model, 'best_model_weights.npz')
```

### 4. 可视化训练过程

在训练完成后，可以通过以下命令绘制训练过程的损失和准确率曲线：

```python
plot_training_progress(train_loss_history, val_loss_history, val_accuracy_history)
```

### 5. 可视化权重和偏置

可以使用以下函数来可视化最佳模型的权重和偏置：

```python
plot_best_model_parameters(best_model)
```

## 模型权重

训练后的模型权重已经上传至以下链接，你可以下载并使用它进行测试：

- [下载模型权重](链接: https://pan.baidu.com/s/1Svfaa2yzIjKIyaw49tK76w 提取码: rt3e)

---

**备注：** 本项目基于 **CIFAR-10** 数据集进行开发，旨在帮助学习神经网络的基本原理和实现方式。
