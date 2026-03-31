# check_and_fix_build.py
# 检查并修复build.py中的get_dataset函数

import os


def check_build_py():
    """检查build.py中的get_dataset函数"""
    print("🔍 检查build.py中的get_dataset函数")
    print("=" * 60)

    build_file = "semilearn/core/utils/build.py"

    if not os.path.exists(build_file):
        print(f"❌ 文件不存在: {build_file}")
        return

    try:
        with open(build_file, 'r') as f:
            content = f.read()

        # 找到get_dataset函数
        get_dataset_start = content.find('def get_dataset')
        if get_dataset_start == -1:
            print("❌ 没有找到get_dataset函数")
            return

        # 提取函数内容（找到下一个def或文件结尾）
        remaining_content = content[get_dataset_start:]
        next_def = remaining_content.find('\ndef ', 1)  # 跳过当前的def
        if next_def != -1:
            function_content = remaining_content[:next_def]
        else:
            function_content = remaining_content

        print("📋 get_dataset函数内容:")
        print("-" * 40)
        lines = function_content.split('\n')
        for i, line in enumerate(lines[:50]):  # 显示前50行
            print(f"{i + 1:2d}: {line}")

        # 检查是否包含camelyon16
        if 'camelyon16' in function_content:
            print("\n✅ 函数中已包含camelyon16处理")
        else:
            print("\n❌ 函数中没有camelyon16处理")
            suggest_build_fix(function_content)

        return function_content

    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None


def suggest_build_fix(function_content):
    """建议build.py的修复方案"""
    print("\n💡 build.py修复建议:")
    print("-" * 40)

    # 查找模式，通常是 if dataset == 'xxx': return get_xxx(...)
    if 'if dataset ==' in function_content:
        print("✅ 发现条件判断模式")
        print("需要添加:")
        print("""
    elif dataset == 'camelyon16':
        return get_camelyon16(args, alg=algorithm, dataset=dataset, 
                             num_labels=num_labels, num_classes=num_classes, 
                             data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)
        """)
    else:
        print("⚠️  没有发现标准的条件判断模式")
        print("需要手动检查函数结构")


def create_build_fix():
    """创建build.py修复脚本"""
    print("\n🔧 创建build.py修复指南")
    print("-" * 40)

    fix_guide = """
# build.py修复指南

1. 打开 semilearn/core/utils/build.py
2. 找到 get_dataset 函数
3. 在函数中添加 camelyon16 的条件判断:

elif dataset == 'camelyon16':
    return get_camelyon16(args, alg=algorithm, dataset=dataset, 
                         num_labels=num_labels, num_classes=num_classes, 
                         data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)

4. 确保在文件顶部导入了 get_camelyon16:
from semilearn.datasets import get_camelyon16

5. 保存文件并重新测试
    """

    with open('build_fix_guide.txt', 'w') as f:
        f.write(fix_guide)

    print("✅ 修复指南已保存: build_fix_guide.txt")


def check_imports_in_build():
    """检查build.py中的导入"""
    print("\n🔍 检查build.py中的导入")
    print("-" * 40)

    build_file = "semilearn/core/utils/build.py"

    try:
        with open(build_file, 'r') as f:
            lines = f.readlines()

        print("📋 导入部分:")
        for i, line in enumerate(lines[:30]):  # 前30行通常包含所有导入
            if 'import' in line or 'from' in line:
                print(f"  {i + 1:2d}: {line.strip()}")

        # 检查是否导入了数据集相关函数
        content = ''.join(lines)
        if 'get_camelyon16' in content:
            print("\n✅ 已导入get_camelyon16")
        else:
            print("\n❌ 没有导入get_camelyon16")
            print("需要添加: from semilearn.datasets import get_camelyon16")

    except Exception as e:
        print(f"❌ 检查导入失败: {e}")


if __name__ == "__main__":
    check_imports_in_build()
    function_content = check_build_py()
    create_build_fix()