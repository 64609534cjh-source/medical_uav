# 在本地发布讨论版：操作清单

此文档不包含远程凭据，也不会自动执行任何push。本轮工具读取仓库返回404；尚未创建远程分支、PR或Release。

## 一、先确认本地状态和目标远端

在你自己的Ubuntu终端执行：

```bash
cd /home/liuyu/ME5418-project/medical_uav
git status --short
git log -5 --oneline
git rev-parse --verify 'd050c4c^{commit}'
git rev-parse HEAD
git remote -v
```

检查当前HEAD与预期版本。工作区不干净、commit不存在或remote不是预期仓库时，先解决差异，不继续后面的命令。查看remote时不要把含凭据的URL复制到公开记录。

使用本机已有Git权限访问远端。ChatGPT连接的404不代表本机Git一定没有权限；如果本机也无写权限，应由仓库owner授权或走fork/PR，不要覆盖别人的远端配置。

## 二、读取远端并扫描历史

```bash
git fetch origin
git branch -r

git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1 == "blob" && $3 > 52428800 {print}'
```

这会列出本地已知Git历史中的大于50MiB的blob，不能代替敏感信息检查。GitHub普通Git会阻止大于100MiB的文件；`.gitignore`不会删除已经进入历史的文件。大检查点/轨迹优先使用Release assets或合适的LFS方案。

如果发现大文件或秘密已在历史中，不要直接force-push，也不要仅新建一个删除文件的commit就以为历史已经清理。先决定安全清理方案并保留源码与产物的溯源；本清单不自动重写历史。

## 三、建立发布分支

仅在基点已确认且分支名尚未存在时执行：

```bash
git switch -c release/discussion-v0.1 d050c4c
mkdir -p docs
```

将本计划中的以下文件复制到仓库的 `docs/`：

- `ROADMAP_GEN4.md`
- `RELEASE_DISCUSSION.md`
- `EXPERIMENT_SPEC.md`
- `PUBLISH_STEPS.md`

这些是新增文档，复制前确认没有同名需保留的文件。然后补充README中的真实运行命令、结果索引和模型manifest。不要复制虚拟环境；不要未经核实编造训练脚本flags。

## 四、核对、提交文档

```bash
git add docs/ROADMAP_GEN4.md docs/RELEASE_DISCUSSION.md \
        docs/EXPERIMENT_SPEC.md docs/PUBLISH_STEPS.md
git diff --cached --stat
git diff --cached
```

检查后提交：

```bash
git commit -m "docs: prepare discussion release and Gen4 research roadmap"
```

若README、manifest等也修改了，请逐个确认后显式 `git add`，不要机械使用 `git add .` 把本机产物全部提交。

## 五、推送讨论分支

确认没有同名冲突，并且本机有权限后：

```bash
git push --dry-run origin HEAD:refs/heads/release/discussion-v0.1
git push -u origin HEAD:refs/heads/release/discussion-v0.1
```

`--dry-run`用于预览ref更新，不保证真实push一定通过所有大小或仓库规则检查。不使用 `--force`。

推送后，在GitHub页面创建Draft PR：

- compare：`release/discussion-v0.1`
- base：仓库实际用于协作的默认/集成分支（先查看，不假定一定是main）
- title：`Baseline discussion release: destination-conditioned PPO and Gen4 roadmap`
- body：使用 `RELEASE_DISCUSSION.md`，补齐实际证据链接。

如果当前发布内容与base没有差异，不能凭空创建PR；改用Issue/Discussion记录审核，并在真实文档差异提交后再开PR。

## 六、审核后打标签与发布

先由另一名组员fresh-clone复现。通过后，在审核通过的发布commit上创建新的annotated tag，例如 `v0.1.0-discussion.1`，然后只推送这个明确标签，再在GitHub创建prerelease。

不要移动历史tag；不要执行不加选择的 `git push --tags`。Release附件应包括配置、所有正式选中模型或其可访问归档、逐日结果摘要、hash清单。旧Test记录保持原名。

标签、模型的训练SHA与本次推理SHA需要分别记录，不要把后来整理文档的SHA误当成训练时源码。

## 七、后续研究

新研究分支从审核通过的基点创建。讨论版只接收明确的复现修复；探索性网络/任务修改在独立PR进行。

## 官方说明

- Draft PR：`https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request`
- Releases：`https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository`
- Git大文件：`https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github`
