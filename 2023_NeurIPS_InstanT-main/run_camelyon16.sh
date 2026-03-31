#!/bin/bash

# run_camelyon16.sh
# CAMELYON16数据集InstanT半监督学习训练脚本 - 完整版
# 策略2：模拟半监督（推荐）

set -e  # 遇到错误立即退出

# ===== 基本参数设置 =====
DATA_DIR="/home/xiaoyuan/Data3/CAMELYON16"
GPU_ID=0
EXPERIMENT_NAME="camelyon16_instant_$(date +%Y%m%d_%H%M%S)"

# ===== 训练参数 =====
BATCH_SIZE=32
LEARNING_RATE=0.0001
EPOCHS=300
NUM_TRAIN_ITER=20000
NUM_LABELS=500  # 每类有标签样本数

# ===== 半监督策略选择 =====
# 'real': 真实半监督 - 少量标注数据 + 大量真实无标签数据
# 'simulated': 模拟半监督（推荐）- 从所有标注数据中抽取部分作为"无标签"数据
SSL_STRATEGY="simulated"

# ===== InstanT算法参数 =====
P_CUTOFF=0.95        # 伪标签置信度阈值
TEMPERATURE=0.5      # 温度参数
WARM_UP_ITER=2000    # 预热迭代次数
ESTIMATION="class"   # 估计方式：'instance' 或 'class'

# ===== 网络和优化器参数 =====
NETWORK="resnet50"   # 网络架构
OPTIMIZER="AdamW"    # 优化器
WEIGHT_DECAY=0.0005  # 权重衰减

echo "=============================================="
echo "  CAMELYON16 InstanT半监督学习训练"
echo "=============================================="
echo "📅 时间: $(date)"
echo "📁 数据路径: $DATA_DIR"
echo "🖥️  GPU: $GPU_ID"
echo "🏷️  实验名称: $EXPERIMENT_NAME"
echo "🎯 半监督策略: $SSL_STRATEGY"
echo "📊 每类标注样本: $NUM_LABELS"
echo "🔄 训练迭代: $NUM_TRAIN_ITER"
echo "=============================================="

# ===== 环境检查 =====
echo "🔍 检查环境..."

# 检查Python和依赖
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装"
    exit 1
fi

# 检查CUDA
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "⚠️  CUDA检查失败，将使用CPU训练"
    GPU_ID="None"
fi

echo "✅ Python环境检查通过"

# ===== 数据检查 =====
echo "🔍 检查数据目录..."

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 错误: 数据根目录不存在: $DATA_DIR"
    echo "请检查数据路径是否正确"
    exit 1
fi

if [ ! -d "$DATA_DIR/Neg_Slide" ]; then
    echo "❌ 错误: 阴性数据目录不存在: $DATA_DIR/Neg_Slide"
    echo "请检查数据结构是否正确"
    exit 1
fi

if [ ! -d "$DATA_DIR/Pos_Slide" ]; then
    echo "❌ 错误: 阳性数据目录不存在: $DATA_DIR/Pos_Slide"
    echo "请检查数据结构是否正确"
    exit 1
fi

# 检查数据文件
NEG_COUNT=$(find "$DATA_DIR/Neg_Slide" -name "*.jpg" | wc -l)
POS_COUNT=$(find "$DATA_DIR/Pos_Slide" -name "*_1.jpg" | wc -l)

echo "✅ 数据目录检查通过"
echo "   📊 阴性样本数: $NEG_COUNT"
echo "   📊 阳性样本数: $POS_COUNT"

if [ $NEG_COUNT -eq 0 ] || [ $POS_COUNT -eq 0 ]; then
    echo "⚠️  警告: 某些类别的样本数量为0，请检查数据"
fi

# ===== 文件检查 =====
echo "🔍 检查必要文件..."

REQUIRED_FILES=("camelyon_ssl.py" "register_camelyon16.py" "camelyon16_instant.yaml" "train.py" "instant.py")

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 缺少必要文件: $file"
        exit 1
    fi
done

echo "✅ 必要文件检查通过"

# ===== 创建必要目录 =====
echo "📁 创建必要目录..."
mkdir -p ./saved_models
mkdir -p ./logs
mkdir -p ./checkpoints

# ===== 注册数据集 =====
echo "📝 注册CAMELYON16数据集..."
python register_camelyon16.py

if [ $? -ne 0 ]; then
    echo "❌ 数据集注册失败，请检查register_camelyon16.py文件"
    exit 1
fi

echo "✅ 数据集注册成功"

# ===== 分析数据集（可选但推荐） =====
echo "📊 分析数据集结构..."
python -c "
import sys
sys.path.append('.')
from camelyon_ssl import analyze_camelyon_dataset
try:
    stats = analyze_camelyon_dataset('$DATA_DIR')
    print(f'建议参数设置:')
    if stats['total_labeled'] >= 2000:
        print(f'  num_labels: 500-1000 (你有 {stats[\"total_labeled\"]} 个标注样本)')
    else:
        print(f'  num_labels: {stats[\"total_labeled\"]//4} (你有 {stats[\"total_labeled\"]} 个标注样本)')
except Exception as e:
    print(f'数据分析失败: {e}')
"

# ===== 开始训练 =====
echo "🚀 开始训练..."
echo "训练日志将保存到: ./logs/${EXPERIMENT_NAME}.log"

# 构建训练命令
TRAIN_CMD="python train.py \
    --c camelyon16_instant.yaml \
    --data_dir '$DATA_DIR' \
    --save_name '$EXPERIMENT_NAME' \
    --gpu $GPU_ID \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --epoch $EPOCHS \
    --num_train_iter $NUM_TRAIN_ITER \
    --num_labels $NUM_LABELS \
    --ssl_strategy '$SSL_STRATEGY' \
    --p_cutoff $P_CUTOFF \
    --T $TEMPERATURE \
    --warm_up_it $WARM_UP_ITER \
    --estimation '$ESTIMATION' \
    --algorithm instant \
    --dataset camelyon16 \
    --net '$NETWORK' \
    --optim '$OPTIMIZER' \
    --weight_decay $WEIGHT_DECAY \
    --num_classes 2 \
    --img_size 224 \
    --use_pretrain \
    --use_tensorboard \
    --overwrite \
    --seed 42"

echo "执行命令: $TRAIN_CMD"
echo ""

# 执行训练并记录日志
eval $TRAIN_CMD 2>&1 | tee "./logs/${EXPERIMENT_NAME}.log"

# 检查训练结果
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 训练成功完成!"
    echo "=============================================="
    echo "📁 结果文件:"
    echo "   📄 日志文件: ./logs/${EXPERIMENT_NAME}.log"
    echo "   💾 模型文件: ./saved_models/${EXPERIMENT_NAME}/"
    echo "   📊 TensorBoard: tensorboard --logdir ./saved_models/${EXPERIMENT_NAME}/tensorboard"
    echo ""
    echo "📈 查看训练曲线:"
    echo "   tensorboard --logdir ./saved_models"
    echo "   然后访问 http://localhost:6006"
    echo ""
    echo "🔍 查看详细日志:"
    echo "   tail -f ./logs/${EXPERIMENT_NAME}.log"
    echo "=============================================="

    # 显示最终结果（如果有的话）
    if [ -f "./logs/${EXPERIMENT_NAME}.log" ]; then
        echo "📊 训练总结:"
        echo "最后几行日志:"
        tail -n 10 "./logs/${EXPERIMENT_NAME}.log" | grep -E "(best|Best|accuracy|Accuracy|auc|AUC)" || true
    fi

else
    echo ""
    echo "❌ 训练失败！退出码: $TRAIN_EXIT_CODE"
    echo "=============================================="
    echo "🔍 错误排查："
    echo "1. 检查日志文件: ./logs/${EXPERIMENT_NAME}.log"
    echo "2. 检查GPU内存是否足够"
    echo "3. 检查数据路径是否正确"
    echo "4. 尝试减小batch_size"
    echo ""
    echo "📄 最近的错误信息:"
    if [ -f "./logs/${EXPERIMENT_NAME}.log" ]; then
        tail -n 20 "./logs/${EXPERIMENT_NAME}.log" | grep -E "(Error|error|Exception|Failed|failed)" || true
    fi
    echo "=============================================="
    exit 1
fi

echo ""
echo "🏁 脚本执行完成"