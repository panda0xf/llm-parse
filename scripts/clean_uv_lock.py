import re
import sys
from pathlib import Path


def clean_uv_lock(lock_file_path: str) -> bool:
    """清理 uv.lock 中的国内镜像源配置。

    移除或重写 `source = { registry = ... }`，
    确保在其他环境和 CI 机器上能使用默认源拉取包。
    """
    path = Path(lock_file_path)
    if not path.exists():
        print(f"Error: Could not find {lock_file_path}")
        return False

    original_content = path.read_text(encoding="utf-8")

    # 将任何 registry URL 替换为不带 registry 指定的样子，或直接替换 URL。
    # 对于 uv.lock 的语法，可以直接将 registry 的值改为 pypi 官方源，
    # 或者对于默认 registry 而言干脆删去这行 (通常没有 registry 设定即为 pypi.org)。

    # 这里我们选择将特定的非官方镜像（如 tuna, aliyun 等）整行删除，让拉取回退到全局设置或官方源
    # 注意: 如果只删除 `source = ...` 行，可能会导致 TOML 表异常，所以推荐替换成官方 registry
    # 或者直接安全地将 registry URL 统统替换为官方源：

    cleaned_content = re.sub(r'registry\s*=\s*"[^"]+"', 'registry = "https://pypi.org/simple"', original_content)

    if original_content != cleaned_content:
        path.write_text(cleaned_content, encoding="utf-8")
        print(f"Cleaned {lock_file_path}: Replaced non-standard registries with PyPI official registry.")
        return True

    return False


if __name__ == "__main__":
    lock_file = sys.argv[1] if len(sys.argv) > 1 else "uv.lock"
    changed = clean_uv_lock(lock_file)
    # pre-commit 会在此命令返回非零时阻止 commit，如果我们修改了文件，我们返回非零状态码
    if changed:
        sys.exit(1)
    sys.exit(0)
