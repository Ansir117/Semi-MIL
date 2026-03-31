# fix_optimizer_error.py
# 移除get_optimizer函数中错误的camelyon16代码

import os


def fix_optimizer_function():
    """移除get_optimizer函数中的camelyon16代码"""
    print("🔧 修复get_optimizer函数中的camelyon16错误")
    print("=" * 60)

    build_file = "semilearn/core/utils/build.py"

    if not os.path.exists(build_file):
        print(f"❌ 文件不存在: {build_file}")
        return False

    # 备份文件
    backup_file = build_file + ".optimizer_backup"
    os.system(f"cp {build_file} {backup_file}")
    print(f"✅ 文件已备份: {backup_file}")

    try:
        with open(build_file, 'r') as f:
            lines = f.readlines()

        print("📋 查找并移除get_optimizer中的camelyon16代码...")

        new_lines = []
        in_optimizer_function = False
        skip_camelyon_block = False
        camelyon_lines_removed = 0

        for i, line in enumerate(lines):
            line_num = i + 1

            # 检测是否进入get_optimizer函数
            if 'def get_optimizer(' in line:
                in_optimizer_function = True
                print(f"  进入get_optimizer函数 (行 {line_num})")
            elif line.strip().startswith('def ') and in_optimizer_function:
                in_optimizer_function = False
                print(f"  离开get_optimizer函数 (行 {line_num})")

            # 在get_optimizer函数中，移除camelyon16相关代码
            if in_optimizer_function and 'camelyon16' in line:
                print(f"  移除行 {line_num}: {line.strip()}")
                skip_camelyon_block = True
                camelyon_lines_removed += 1
                continue
            elif skip_camelyon_block:
                # 继续跳过camelyon16块的后续行，直到遇到新的elif/else或函数结束
                if (line.strip() == '' or
                        line.strip().startswith('elif ') or
                        line.strip().startswith('else:') or
                        line.strip().startswith('def ') or
                        not line.startswith('    ')):  # 不再是缩进的代码块
                    skip_camelyon_block = False
                    # 如果是空行或者新的条件，不跳过这一行
                    if not (line.strip() == '' and in_optimizer_function):
                        new_lines.append(line)
                else:
                    print(f"  移除行 {line_num}: {line.strip()}")
                    camelyon_lines_removed += 1
                    continue
            else:
                new_lines.append(line)

        print(f"📊 移除了 {camelyon_lines_removed} 行camelyon16相关代码")

        # 写回文件
        with open(build_file, 'w') as f:
            f.writelines(new_lines)

        print("✅ get_optimizer函数已修复")
        return True

    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def ensure_camelyon16_in_get_dataset():
    """确保camelyon16代码在get_dataset函数中"""
    print("\n🔍 确保get_dataset函数中有camelyon16处理")
    print("-" * 50)

    build_file = "semilearn/core/utils/build.py"

    try:
        with open(build_file, 'r') as f:
            content = f.read()

        # 查找get_dataset函数
        get_dataset_start = content.find('def get_dataset(')
        if get_dataset_start == -1:
            print("❌ 找不到get_dataset函数")
            return False

        # 提取get_dataset函数内容
        remaining_content = content[get_dataset_start:]
        next_def = remaining_content.find('\ndef ', 1)
        if next_def != -1:
            get_dataset_content = remaining_content[:next_def]
        else:
            get_dataset_content = remaining_content

        # 检查是否包含camelyon16处理
        if 'camelyon16' in get_dataset_content:
            print("✅ get_dataset函数中已有camelyon16处理")

            # 检查导入是否正确
            if 'get_camelyon16' in content:
                print("✅ get_camelyon16已导入")
            else:
                print("⚠️  需要导入get_camelyon16")
                add_camelyon16_import()

            return True
        else:
            print("❌ get_dataset函数中缺少camelyon16处理")
            add_camelyon16_to_get_dataset()
            return True

    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def add_camelyon16_import():
    """添加get_camelyon16导入"""
    print("🔧 添加get_camelyon16导入...")

    build_file = "semilearn/core/utils/build.py"

    try:
        with open(build_file, 'r') as f:
            content = f.read()

        # 查找导入get_camelyon16的位置
        import_line = "from semilearn.datasets import get_eurosat, get_medmnist, get_semi_aves, get_cifar, get_svhn, get_stl10, get_imagenet, get_json_dset, get_pkl_dset"

        if import_line in content:
            new_import_line = import_line + ", get_camelyon16"
            content = content.replace(import_line, new_import_line)

            with open(build_file, 'w') as f:
                f.write(content)

            print("✅ get_camelyon16导入已添加")
        else:
            print("⚠️  找不到标准导入行，需要手动添加")

    except Exception as e:
        print(f"❌ 添加导入失败: {e}")


def add_camelyon16_to_get_dataset():
    """在get_dataset函数中添加camelyon16处理"""
    print("🔧 在get_dataset函数中添加camelyon16处理...")

    build_file = "semilearn/core/utils/build.py"

    try:
        with open(build_file, 'r') as f:
            lines = f.readlines()

        # 找到get_dataset函数中的else语句并在之前插入
        new_lines = []
        in_get_dataset = False
        camelyon16_added = False

        for i, line in enumerate(lines):
            if 'def get_dataset(' in line:
                in_get_dataset = True
            elif line.strip().startswith('def ') and in_get_dataset:
                in_get_dataset = False

            # 在get_dataset函数的else语句之前插入camelyon16处理
            if (in_get_dataset and
                    line.strip() == 'else:' and
                    not camelyon16_added and
                    i + 1 < len(lines) and
                    'return None' in lines[i + 1]):
                # 插入camelyon16处理
                camelyon16_code = [
                    '    elif dataset == "camelyon16":\n',
                    '        lb_dset, ulb_dset, eval_dset = get_camelyon16(args, alg=algorithm, dataset=dataset, num_labels=num_labels, num_classes=num_classes, data_dir=data_dir, include_lb_to_ulb=include_lb_to_ulb)\n',
                    '        test_dset = None\n'
                ]

                new_lines.extend(camelyon16_code)
                camelyon16_added = True
                print("✅ camelyon16处理已添加到get_dataset函数")

            new_lines.append(line)

        # 写回文件
        with open(build_file, 'w') as f:
            f.writelines(new_lines)

        if camelyon16_added:
            print("✅ get_dataset函数已更新")
        else:
            print("⚠️  未找到合适的插入位置")

    except Exception as e:
        print(f"❌ 添加失败: {e}")


def verify_complete_fix():
    """验证完整修复"""
    print("\n🔍 验证完整修复结果")
    print("-" * 40)

    build_file = "semilearn/core/utils/build.py"
    success = True

    try:
        with open(build_file, 'r') as f:
            content = f.read()

        # 1. 检查get_optimizer中不应该有camelyon16
        get_optimizer_start = content.find('def get_optimizer(')
        if get_optimizer_start != -1:
            remaining = content[get_optimizer_start:]
            next_def = remaining.find('\ndef ', 1)
            if next_def != -1:
                optimizer_content = remaining[:next_def]
            else:
                optimizer_content = remaining[:1000]  # 假设函数不会太长

            if 'camelyon16' in optimizer_content:
                print("❌ get_optimizer中仍有camelyon16代码")
                success = False
            else:
                print("✅ get_optimizer函数已清理")

        # 2. 检查get_dataset中应该有camelyon16
        get_dataset_start = content.find('def get_dataset(')
        if get_dataset_start != -1:
            dataset_remaining = content[get_dataset_start:]
            if 'camelyon16' in dataset_remaining:
                print("✅ get_dataset函数中有camelyon16处理")
            else:
                print("❌ get_dataset函数中缺少camelyon16处理")
                success = False

        # 3. 检查导入
        if 'get_camelyon16' in content:
            print("✅ get_camelyon16已导入")
        else:
            print("❌ 缺少get_camelyon16导入")
            success = False

        return success

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


if __name__ == "__main__":
    print("🔧 开始完整修复build.py中的camelyon16问题...")

    # 步骤1: 从get_optimizer中移除camelyon16
    if fix_optimizer_function():
        # 步骤2: 确保get_dataset中有camelyon16
        ensure_camelyon16_in_get_dataset()

        # 步骤3: 验证修复
        if verify_complete_fix():
            print("\n🎉 所有修复完成！")
            print("📋 下一步:")
            print("1. 重新运行: python debug_training_init.py")
            print("2. 如果成功，开始训练: python train.py --c config/classic_cv/instant/instant_camelyon16.yaml")
        else:
            print("\n⚠️  部分修复可能不完整，建议手动检查")
    else:
        print("\n❌ 修复失败，需要手动处理")