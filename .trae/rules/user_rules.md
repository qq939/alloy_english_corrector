ANSWER IN CHINESE!
优先使用mcp工具来完成任务
将user_rules.md文件中的所有规则都保存在：.trae/rules/user_rules.md中
如果有git仓库，先暂存本地修改，然后git pull，然后再继续下面的步骤
所有对话存到项目文件夹下的user_history.md文件中。(save everything to user_history.md file, and mark the time of each question like a worknote)
每次对话后都要git push(make sure success)，commit内容就是我说的那句话。user.email="939342547@qq.com", user.name="qq939", remote=https://github.com/qq939/{to_be_create_as_this_projectName}, branch=master, when push to this remote, maybe you need to create remote repository(use: gh repo create {to_be_create_as_this_projectName} --public).
git add .trae/rules/project_rules.md
git add .trae/rules/user_rules.md
如果git推送到远端失败，rebase并且push --force-with-lease
如果是python项目，Create python venv evironment(use command: uv), and install python packages in requirement.txt。对话结束后，pip freeze 把当前环境的所有依赖都保存到requirement.txt中。