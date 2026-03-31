#!/bin/bash
# fix_dataset_registration.sh
# 修复camelyon16数据集注册

echo "🔧 修复camelyon16数据集注册..."

# 备份原文件
cp semilearn/datasets/cv_datasets/__init__.py semilearn/datasets/cv_datasets/__init__.py.backup

echo "✅ 原文件已备份"

# 检查是否需要添加导入
if ! grep -q "from .camelyon16 import get_camelyon16" semilearn/datasets/cv_datasets/__init__.py; then
    echo "添加导入语句..."
    # 在其他导入语句后添加
    sed -i '/from \.cifar import get_cifar/a from .camelyon16 import get_camelyon16' semilearn/datasets/cv_datasets/__init__.py
    echo "✅ 导入语句已添加"
else
    echo "✅ 导入语句已存在"
fi

echo "🎯 接下来需要手动添加get_dataset函数中的条件判断"
echo "请在get_dataset函数中添加camelyon16的处理逻辑"
