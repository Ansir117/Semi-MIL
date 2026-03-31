#!/bin/bash

# create_instant_env.sh
# 创建InstanT半监督学习专用环境

set -e  # 遇到错误立即退出

echo "🚀 创建InstanT半监督学习专用环境"
echo "=================================="

# 环境名称
ENV_NAME="instant_ssl"
PYTHON_VERSION="3.8"

echo "📝 环境配置:"
echo "   环境名称: $ENV_NAME"
echo "   Python版本: $PYTHON_VERSION"
echo "   目标GPU: RTX 3090"
echo "=================================="

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装或不在PATH中"
    echo "请确保已安装Anaconda或Miniconda"
    exit 1
fi

# 删除同名环境（如果存在）
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  环境 '$ENV_NAME' 已存在，是否删除重建？ (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🗑️  删除现有环境..."
        conda env remove -n $ENV_NAME -y
    else
        echo "❌ 取消操作"
        exit 1
    fi
fi

# 创建新环境
echo "🔨 创建新环境: $ENV_NAME"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 激活环境
echo "🔄 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 验证环境
echo "✅ 环境创建成功:"
echo "   Python: $(python --version)"
echo "   环境路径: $(which python)"

echo ""
echo "🎉 环境创建完成！"
echo "=================================="
echo "📋 下一步操作:"
echo "1. 激活环境: conda activate $ENV_NAME"
echo "2. 运行依赖安装脚本: ./install_dependencies.sh"
echo "=================================="