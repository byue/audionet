
CURR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
STALE_BRANCH=$(git branch -vv | grep -o "\[origin/$CURR_BRANCH\: gone\]")
TRACKED_BRANCH=$(git branch -vv | grep -o "\[origin/$CURR_BRANCH\]")

function stale_branch_error {
    /bin/echo -e "\e[1;31m************************************\e[0m"
    /bin/echo -e "\e[1;31m*              ERROR               *\e[0m"
    /bin/echo -e "\e[1;31m************************************\e[0m"
    /bin/echo -e "\e[1;31mYour current branch, $CURR_BRANCH, is stale and has already been merged into dev.\e[0m"
    /bin/echo -e "\e[1;31mCreate a new feature branch from dev to perform any new development.\e[0m"
}

function protected_branch_error {
    /bin/echo -e "\e[1;31m************************************\e[0m"
    /bin/echo -e "\e[1;31m*              ERROR               *\e[0m"
    /bin/echo -e "\e[1;31m************************************\e[0m"
    /bin/echo -e "\e[1;31mYour current branch, $CURR_BRANCH, is a protected branch.\e[0m"
    /bin/echo -e "\e[1;31mCreate a new feature branch from dev to perform any new development.\e[0m"
}

function delete_stale_branches {
    git remote prune origin &> /dev/null
    git branch -vv | grep 'origin/.*: gone]' | awk '{print $1}' | xargs git branch -D
}

function configure_remote_tracking {
    delete_stale_branches
    git branch -vv | tr -d '*' | awk '{print $1}' | xargs -I {} git checkout {} && git pull origin {} && git push --set-upstream origin {}
    git checkout $CURR_BRANCH
}
