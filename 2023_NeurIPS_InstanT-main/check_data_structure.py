# check_data_structure.py
# 详细检查CAMELYON16数据目录结构

import os


def check_directory_structure(data_dir, max_depth=3, current_depth=0):
    """递归检查目录结构"""
    if current_depth > max_depth:
        return

    try:
        items = os.listdir(data_dir)
        indent = "  " * current_depth

        for item in sorted(items):
            item_path = os.path.join(data_dir, item)

            if os.path.isdir(item_path):
                try:
                    subitem_count = len(os.listdir(item_path))
                    print(f"{indent}📁 {item}/ ({subitem_count} items)")

                    # 如果是可能的类别目录，显示更多细节
                    if item in ['0', '1', 'normal', 'tumor', 'positive', 'negative']:
                        print(f"{indent}  ↳ 这可能是类别目录")
                        subindent = indent + "    "
                        try:
                            subitems = os.listdir(item_path)[:5]  # 只显示前5个
                            for subitem in subitems:
                                subitem_path = os.path.join(item_path, subitem)
                                if os.path.isfile(subitem_path):
                                    size_mb = os.path.getsize(subitem_path) / (1024 * 1024)
                                    print(f"{subindent}📄 {subitem} ({size_mb:.1f}MB)")
                            if len(os.listdir(item_path)) > 5:
                                print(f"{subindent}... 还有 {len(os.listdir(item_path)) - 5} 个文件")
                        except:
                            pass

                    # 递归检查子目录（但限制深度）
                    if current_depth < max_depth:
                        check_directory_structure(item_path, max_depth, current_depth + 1)

                except PermissionError:
                    print(f"{indent}📁 {item}/ (权限不足)")
                except Exception as e:
                    print(f"{indent}📁 {item}/ (错误: {e})")
            else:
                try:
                    size_mb = os.path.getsize(item_path) / (1024 * 1024)
                    print(f"{indent}📄 {item} ({size_mb:.1f}MB)")
                except:
                    print(f"{indent}📄 {item}")

    except Exception as e:
        print(f"读取目录失败: {e}")


def main():
    """主检查函数"""
    data_dir = "/home/xiaoyuan/Data3/CAMELYON16"

    print("🔍 详细检查CAMELYON16数据目录结构")
    print("=" * 60)
    print(f"📁 根目录: {data_dir}")
    print("-" * 60)

    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return

    check_directory_structure(data_dir, max_depth=2)

    print("\n" + "=" * 60)
    print("🎯 根据上述结构，我们来分析可能的训练数据位置:")
    print("1. 寻找包含图片文件的目录")
    print("2. 寻找可能的类别目录 (0/1, normal/tumor, positive/negative)")
    print("3. 寻找train/test/valid等标准目录")


if __name__ == "__main__":
    main()