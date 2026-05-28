# Blog - Hugo Static Site

## 项目概述

基于Hugo的静态博客，使用PaperModX主题，部署至GitHub Pages（`/docs`目录）。

## 技术栈

- **框架**: Hugo
- **主题**: PaperModX (git submodule)
- **部署**: GitHub Pages via `/docs` directory
- **数学公式**: LaTeX (goldmark passthrough)
- **搜索**: Fuse.js (JSON-based)
- **代码高亮**: Highlight.js (darcula风格)

## 目录结构

```
blog/
├── archetypes/         # 内容模板
│   └── default.md
├── assets/             # 主题资源
├── content/            # 博客内容
│   ├── archives.md     # 归档页
│   ├── search.md       # 搜索页
│   ├── open_courses/   # 学习课程笔记
│   ├── posts/          # 博客文章
│   │   ├── aiinfra/    # AI基础设施
│   │   ├── llmtheory/  # LLM理论
│   │   └── programming/# 编程
│   └── social/         # 社交相关
├── layouts/            # 自定义布局覆盖
├── static/            # 静态资源
├── themes/papermod/    # 主题 (submodule)
├── config.yml         # Hugo配置
└── deploy.sh          # 部署脚本
```

## 主要分类 (menu)

| 名称 | URL | 说明 |
|------|-----|------|
| Archive | /archives/ | 文章归档 |
| Search | /search/ | 搜索页面 |
| AI Infra | /posts/aiinfra/ | AI基础设施文章 |
| Theory | /posts/llmtheory/ | LLM理论文章 |
| Social | /social/ | 社交相关 |
| Programming | /posts/programming/ | 编程文章 |
| Study | /open_courses/ | 公开课笔记 |

## 常用命令

```bash
# 本地预览
hugo server

# 构建并部署
./deploy.sh "commit message"

# 仅构建
hugo --gc --minify -d docs
```

## front matter模板

```yaml
+++
date = '{{ .Date }}'
draft = true
title = '{{ replace .File.ContentBaseName "-" " " | title }}'
+++
```

## 注意事项

- 主题通过`.gitmodules`管理，更新主题需在`themes/papermod`目录操作
- 部署脚本使用`zsh`，且push到远程`hyx`remote的`master`分支
- `config.yml`中配置了LaTeX支持（block: `$$...$$`, inline: `$...$`）
- 站点名称: **CctoctoFX** (Yuxiao Huang)
