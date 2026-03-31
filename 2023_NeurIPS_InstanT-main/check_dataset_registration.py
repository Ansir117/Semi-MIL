# check_dataset_registration.py
# 检查USB框架中的数据集注册情况

import os


def check_init_file():
    """检查__init__.py文件内容"""
    print("🔍 检查数据集注册文件")
    print("=" * 50)

    init_file = "semilearn/datasets/cv_datasets/__init__.py"

    if not os.path.exists(init_file):
        print(f"❌ 文件不存在: {init_file}")
        return

    print(f"📄 读取文件: {init_file}")

    try:
        with open(init_file, 'r') as f:
            content = f.read()

        print(f"✅ 文件读取成功，总长度: {len(content)} 字符")

        # 检查导入部分
        print("\n📋 导入部分:")
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('from .') or line.strip().startswith('import'):
                print(f"  {i + 1:3d}: {line}")

        # 检查camelyon16相关内容
        print(f"\n🔍 查找camelyon16相关内容:")
        camelyon_lines = []
        for i, line in enumerate(lines):
            if 'camelyon16' in line.lower():
                camelyon_lines.append((i + 1, line))

        if camelyon_lines:
            print("✅ 找到camelyon16相关内容:")
            for line_num, line in camelyon_lines:
                print(f"  {line_num:3d}: {line}")
        else:
            print("❌ 没有找到camelyon16相关内容")

        # 查找get_dataset函数
        print(f"\n🔍 查找get_dataset函数:")
        get_dataset_start = content.find('def get_dataset')
        if get_dataset_start != -1:
            print("✅ 找到get_dataset函数")

            # 提取get_dataset函数的主要部分
            function_content = content[get_dataset_start:get_dataset_start + 2000]  # 前2000字符
            print("\n📋 get_dataset函数内容（部分）:")
            function_lines = function_content.split('\n')[:30]  # 前30行
            for i, line in enumerate(function_lines):
                print(f"  {i + 1:2d}: {line}")
        else:
            print("❌ 没有找到get_dataset函数")

        return content

    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None


def suggest_fix(content):
    """建议修复方案"""
    print(f"\n💡 修复建议:")
    print("-" * 40)

    if content is None:
        print("❌ 无法提供建议，文件读取失败")
        return

    # 检查是否已经导入
    if 'from .camelyon16 import get_camelyon16' in content:
        print("✅ camelyon16已经导入")
    else:
        print("❌ 需要添加导入语句:")
        print("   在导入部分添加: from .camelyon16 import get_camelyon16")

    # 检查是否在get_dataset中处理
    if 'camelyon16' in content and 'get_camelyon16' in content:
        print("✅ get_dataset函数中可能已经处理camelyon16")
    else:
        print("❌ 需要在get_dataset函数中添加camelyon16处理:")
        print("""
   在get_dataset函数中添加:

   if dataset == 'camelyon16':
       return get_camelyon16(args, alg=algorithm, dataset=dataset, 
                            num_labels=num_labels, num_classes=num_classes, 
                            data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        """)


def create_fix_script():
    """创建修复脚本"""
    print(f"\n🔧 创建修复脚本")
    print("-" * 40)

    fix_script = """#!/bin/bash
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
"""

    with open('fix_dataset_registration.sh', 'w') as f:
        f.write(fix_script)

    print("✅ 修复脚本已创建: fix_dataset_registration.sh")
    print("运行: chmod +x fix_dataset_registration.sh && ./fix_dataset_registration.sh")


if __name__ == "__main__":
    content = check_init_file()
    suggest_fix(content)
    create_fix_script()