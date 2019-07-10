#!/bin/bash
git remote prune origin &> /dev/null
source ./.git/hooks/common.sh

# stale branch encountered
if [ ! -z "$STALE_BRANCH" ]
then
    stale_branch_error
    exit 1
fi

# pull down latest changes
if [ "$CURR_BRANCH" = "dev" ]
then
    git pull
fi

# active branch with no tracking encountered
if [ -z "$TRACKED_BRANCH" ]
then
    git push --set-upstream origin "$CURR_BRANCH"
fi
