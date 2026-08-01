# MathModel_shared — 共享仓

> 数学建模竞赛团队的协作仓库：论文库、赛题数据、LaTeX 模板、表达库、数学讲解 skill，以及队友间的共享资源。
>
> **本仓库不是独立的竞赛项目**，它是队长项目（核心仓）的共享部分 + 团队的资源中转站。你只需要知道怎么用这个仓库，不需要了解核心仓。

---

## 一、快速开始（第一次使用，5 步）

1. **注册 GitHub**：打开 [github.com](https://github.com)，免费注册一个账号
2. **把账号发给队长**：队长把你加进仓库的协作者名单（队长操作，你只需要把 GitHub 用户名发给他）。不加入也能看（仓库是公开的），但**只有协作者才能上传**
3. **克隆仓库**（在你电脑上任意位置执行）：
   ```
   git clone https://github.com/Clmzz-gra/MathModel_shared.git
   ```
4. **进入目录**：
   ```
   cd MathModel_shared
   ```
5. 完成 ✅ 以后都在这个文件夹里工作。

> 没有 git 的电脑需要先安装 Git：下载 https://git-scm.com/download/win 安装，一路"Next"即可。

---

## 二、日常协作（记住 4 条命令）

每次开工、每次改完，都围绕这 4 条命令：

```bash
git pull          # 开工前必做：把别人最新的改动拉下来
git add .         # 把本次所有改动"放入待提交区"
git commit -m "写清楚改了什么"   # 提交（-m 后面是说明文字）
git push          # 上传到 GitHub，别人就能看到了
```

**黄金流程**：`pull → 改 → add → commit → push`。改任何东西之前先 `pull`，避免覆盖队友的工作。

---

## 三、目录说明 —— 共享部分

这个仓库里有一部分目录是**共享部分**（会与队长的核心仓自动同步，是项目正式资产）：

| 目录 | 内容 |
|------|------|
| `references/` | 历年论文 + 解析笔记（参考论文库） |
| `problems/` | 历年赛题原始数据（PDF + 附件） |
| `latex/` | LaTeX 论文模板与样式 |
| `code-gems/` | 教学级代码范例 |
| `knowledge/expression-library.md` | **表达库**：写作句式积累（⚠️ 受保护，见第五节） |
| `.trae/skills/math/` | 数学讲解 skill（队长单向维护，⚠️ 只读使用，不要修改） |

### 资源区（`resources/`，队友专用）

**`resources/` 是队友的共享地盘**：不受 PR 限制，可以自由创建子目录、自由修改，直接 push。想分享任何东西，一个资源 = 一个子目录放进去：

```
resources/2024-B优秀论文/
resources/常用代码片段/
```

每个子目录放一个 `说明.md`（内容 / 用途 / 来源 / 上传人），commit 写清楚（如 `资源: 添加 xxx`）。详细规范见 [resources/README.md](resources/README.md)。

资源区的内容**只在共享仓内流通，不会同步到队长的核心仓**。

| 区域 | 规则 |
|------|------|
| 共享部分（上表目录） | 正式资产，会双向同步，改动需走 PR |
| 资源区（`resources/` 及子目录） | 随意存放、随意分享、直接 push；不影响共享部分 |

> `.trae/skills/math/` 例外：虽然列在共享部分，但它是**队长贡献的资源**，只供你们使用（复制到自己项目），**不要修改或 push 改动**。

**规则**：临时资源、参考材料统一放 **`resources/`**；**不要把临时文件塞进 `references/` 等共享目录**——那会被同步走且受 PR 保护。

---

## 四、示例：把一篇论文笔记分享出去

```bash
git pull                              # 1. 拉最新
mkdir resources/论文笔记                # 2. 在资源区建目录
# 把论文笔记文件放进去（拖拽/复制）
git add .                             # 3. 暂存
git commit -m "添加 XXX 论文笔记"      # 4. 提交
git push                              # 5. 上传
```

---

## 五、受保护内容（只能通过 Pull Request 修改）

以下共享部分**不能直接 push**，必须走 **Pull Request（PR）** 流程（由队长审查后合并）：

- `references/`（历年论文库）
- `problems/`（历年赛题原始数据）
- `latex/`（LaTeX 模板）
- `code-gems/`（教学代码）
- `knowledge/expression-library.md`（表达库——人类维护的写作资产）

> 直接 push 以上路径会被 GitHub 拒绝，这是正常现象。`resources/` 和 `.trae/skills/math/` 不受此限制。

### 修改受保护文件的 PR 流程

```bash
git pull                                            # 1. 拉最新
git checkout -b fix-expression                      # 2. 新建分支（名字随意，如 fix-expression）
# 修改文件 ...
git add .
git commit -m "更新表达库：xxx"                     # 3. 提交到分支
git push origin fix-expression                      # 4. 推送分支（不是直接推 main）
```

5. 打开 GitHub 仓库页面，会看到一个黄色横幅 **"Compare & pull request"** → 点击它
6. 写清楚改了什么 → 点 **Create pull request**
7. 等队长审查并合并 ✅

---

## 六、常见问题（FAQ）

| 问题 | 原因 | 解决 |
|------|------|------|
| push 被拒绝（non-fast-forward） | 别人先改了，你的本地落后 | 先 `git pull`，再 `git push` |
| 出现冲突（CONFLICT） | 你和别人改了同一处 | `git pull` 后手动解决冲突文件（删除 `<<<<<<<` `=======` `>>>>>>>` 标记，保留正确内容），然后 `git add` + `git commit` + `git push` |
| push 被拒（protected / 403） | 你改的是受保护文件 | 按第五节走 PR 流程 |
| 文件太大传不上去 | GitHub 限制单文件 ≤100MB（>50MB 警告） | 大文件不要传仓库，用网盘分享 |
| 误传了不该传的文件 | — | 立即告诉队长处理（已在 git 历史里的文件需要额外操作） |

**禁忌**：
- 不要传任何 >50MB 的文件
- 不要传证件、密码、私密资料
- 不要 push 前不 pull

---

## 七、约定

- **表达库**全部由人类维护，禁止 AI 修改
- commit 信息写清楚改了什么（如 `添加 2018B 论文笔记`，而不是 `update`）
- 改共享部分前先 pull；冲突时优先沟通再解决
