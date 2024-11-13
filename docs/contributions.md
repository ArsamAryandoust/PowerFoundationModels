# Contributions

We use one long-lived branch for this project, namely "master".


```
-------------------------------------master-------------------------------------
```


In order to get started with contributing code, follow these steps:
  1. `git clone https://github.com/ArsamAryandoust/AI4Climate`
  or `git clone https://<your_personal_access_token>@github.com/ArsamAryandoust/AI4Climate`
  2. `git branch <your_personal_branch>`
  3. `git checkout <your_personal_branch>`


```
-------------------------------------master-------------------------------------
            ----------------<your_personal_branch>----------------
```

All changes that you make should be done to <your_personal_branch>. In a running workflow, where others from the team are contributing to the project simultaneously, you should always make sure that your code has no collision with the latest changes to the "master" branch, before opening a pull request. For this, please follow these steps:
  1. `git checkout <your_personal_branch>`
  2. `git fetch origin naster:naster`
  3. `git merge master` (resolve conflicts locally, if they occur)
  4. `git push -u origin <your_personal_branch>`
  5. On the remote host, create a pull request for <your_personal_branch> into the "master" branch 


While making changes to <your_personal_branch>, you can create arbitrary short-lived branches named <your_feature_branch> originating from <your_personal_branch>. Make sure you test and merge/rebase these smaller feature branches before pushing <your_personal_branch> to the remote repository and openning a pull request.


```
-------------------------------------master-------------------------------------
            ----------------<your_personal_branch>----------------
                     -------<your_feature_branch>-------
```

Warning notice:

Do NOT use Rebase on commits that you have already pushed to a remote repository! Instead, use Rebase only for cleaning up your local commit history before merging it into a shared team branch.


