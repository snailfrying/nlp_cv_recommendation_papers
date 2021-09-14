#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import urllib.parse as up

paper_class_map = {}
paper_map = {}

file_object = open('./README.md', encoding='utf-8')
all_lines = file_object.readlines()
file_object.close()

# 可直接追加，输入到历史readme文件中
out_file = open('./README1.md', 'w', encoding='utf-8')

github_root = "https://github.com/snailfrying/nlp_cv_recommendation_papers/blob/main/"

local_dirs = os.listdir("./")
# 直接指定生成文件
push_files = ['cv_papers', 'nlp_papers', 'recommend_papers', 'optimization_papers']

# 获取github仓库里pdf的链接，我这有两层目录
def git_github_papers_pdf(github_root, local_dirs, push_files):
    for one_dir in local_dirs:
        # 过滤掉特定文件
        if one_dir == 'Image':
            continue
        # 判断是否有此文件，并且把目录名，作为第一标题
        if os.path.isdir(one_dir) and not one_dir.startswith('.'):
            out_file.write("\n### " + one_dir + "\n")
            if one_dir.strip() in paper_class_map:
                out_file.write(paper_class_map[one_dir.strip()] + "\n")
            dirs = os.listdir(one_dir)
            # 第二层目录
            for files in dirs:
                out_file.write("\n**" + files + "**\n")
                # 获取第二层目录文件，即pdf
                files1 = os.listdir(os.path.join('./', one_dir, files))
                for one_file in files1:
                    if not os.path.isdir(one_file) and not one_file.startswith('.'):
                        # 获paper url
                        pdf_name = "* [" + ('.').join(one_file.split('.')[:-1]) + "](" + github_root + up.quote(
                            one_dir.strip()) + "/" + up.quote(files.strip()) + "/" \
                                   + up.quote(one_file.strip()) + ") <br />\n"
                        out_file.write(pdf_name)
                        if one_file.strip() in paper_map:
                            out_file.write(paper_map[one_file.strip()] + "\n")

    out_file.close()


if __name__ == '__main__':
    git_github_papers_pdf(github_root, push_files, push_files)
