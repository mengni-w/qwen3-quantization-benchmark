#!/usr/bin/env zsh
# sync_to_github.sh
# 用法:
#   ./scripts/sync_to_github.sh                # 交互式，引导你选择提供远程 URL 或使用 gh 创建仓库
#   ./scripts/sync_to_github.sh --remote URL  # 使用指定的远程仓库 URL
#   ./scripts/sync_to_github.sh --create NAME # 使用 gh CLI 创建仓库（必须已登录 gh）

set -euo pipefail

# 找到仓库根目录（假设脚本位于 scripts/ 下）
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "仓库根目录: $REPO_ROOT"

# 检查 git
if ! command -v git >/dev/null 2>&1; then
  echo "错误: 未找到 git，请先安装 git。"
  exit 1
fi

# 参数解析（简单）
REMOTE_URL=""
CREATE_NAME=""
if [[ ${1:-} == "--remote" ]]; then
  REMOTE_URL="$2"
elif [[ ${1:-} == "--create" ]]; then
  CREATE_NAME="$2"
fi

# 如果不是 git 仓库则初始化
if [[ ! -d .git ]]; then
  echo "未检测到 .git，正在初始化 git 仓库..."
  git init
  # 创建一个基础 .gitignore（若不存在）
  if [[ ! -f .gitignore ]]; then
    cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
.env
.venv/

# macOS
.DS_Store

# IDE
.vscode/
.idea/

# Results
results/

# Logs
*.log

EOF
    echo ".gitignore 已创建"
  fi
else
  echo "检测到现有 git 仓库"
fi

# 若没提交则创建一次初始提交（添加所有文件）
if [[ -z "$(git rev-parse --verify HEAD 2>/dev/null || true)" ]]; then
  echo "仓库没有提交，创建初始提交..."
  git add -A
  git commit -m "Initial commit"
else
  echo "仓库已有提交"
fi

# Helper: 当前分支（若 HEAD 不存在，创建 main）
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [[ -z "$CURRENT_BRANCH" || "$CURRENT_BRANCH" == "HEAD" ]]; then
  git checkout -b main
  CURRENT_BRANCH="main"
fi

# 优先处理命令行参数
if [[ -n "$REMOTE_URL" ]]; then
  echo "使用提供的远程: $REMOTE_URL"
  if git remote | grep -q '^origin$'; then
    echo "已有 origin，更新 origin 为 $REMOTE_URL"
    git remote set-url origin "$REMOTE_URL"
  else
    git remote add origin "$REMOTE_URL"
  fi
elif [[ -n "$CREATE_NAME" ]]; then
  # 使用 gh 创建仓库
  if ! command -v gh >/dev/null 2>&1; then
    echo "错误: 未安装 gh CLI，无法自动创建仓库。请安装并登录 gh，或改用 --remote 提供远程 URL。"
    exit 1
  fi
  echo "使用 gh 创建仓库: $CREATE_NAME"
  # gh repo create <name> --source=. --public --push
  gh repo create "$CREATE_NAME" --source=. --public --push
  echo "远程已创建并已 push（如果 gh 成功执行）"
  exit 0
else
  # 交互式询问
  if ! git remote | grep -q '^origin$'; then
    echo
    echo "请选择如何设置远程仓库："
    echo "  1) 提供远程 URL（例如 git@github.com:你/仓库.git）"
    echo "  2) 使用 gh CLI 在 GitHub 上创建仓库并 push（需要已安装并登录 gh）"
    echo "  3) 取消"
    echo -n "输入 1/2/3: "
    read -r CHOICE
    if [[ "$CHOICE" == "1" ]]; then
      echo -n "请输入远程仓库 URL: "
      read -r REMOTE_URL
      if [[ -z "$REMOTE_URL" ]]; then
        echo "未输入 URL，退出。"
        exit 1
      fi
      git remote add origin "$REMOTE_URL"
    elif [[ "$CHOICE" == "2" ]]; then
      if ! command -v gh >/dev/null 2>&1; then
        echo "错误: 未安装 gh CLI，无法自动创建仓库。"
        exit 1
      fi
      echo -n "请输入要创建的仓库名（例如 my-repo，使用你的 GitHub 账户或 org）: "
      read -r CREATE_NAME
      if [[ -z "$CREATE_NAME" ]]; then
        echo "未输入仓库名，退出。"
        exit 1
      fi
      gh repo create "$CREATE_NAME" --source=. --public --push
      echo "远程已创建并已 push（如果 gh 成功执行）"
      exit 0
    else
      echo "已取消。"
      exit 0
    fi
  else
    echo "已存在 origin 远程：$(git remote get-url origin)"
  fi
fi

# 到这里应该有 origin
if ! git remote | grep -q '^origin$'; then
  echo "错误：未设置 origin 远程。请手动添加或使用 --create/--remote 参数。"
  exit 1
fi

# 确保本地分支命名为 main（安全重命名）
if [[ "$CURRENT_BRANCH" != "main" ]]; then
  echo "将当前分支 $CURRENT_BRANCH 重命名为 main 并切换（如果你不希望这样，请先手动处理）"
  git branch -M main || true
fi

# 推送到远程
echo "推送到 origin main..."
# 首次推送使用 -u
git push -u origin main

echo "同步完成。远程仓库地址：$(git remote get-url origin)"

echo "如果你想在 GitHub 上打开仓库页面，可以运行："
echo "  gh repo view --web"

exit 0
