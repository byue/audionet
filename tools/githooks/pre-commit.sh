#!/bin/bash
git remote prune origin &> /dev/null
source ./.git/hooks/common.sh

# prevent commits to protected branches
if [ "$CURR_BRANCH" = "dev" ] || [ "$CURR_BRANCH" = "master" ] || [ "$CURR_BRANCH" = "deploy" ]
then
    protected_branch_error
    exit 1
fi

# stale branch encountered
if [ ! -z "$STALE_BRANCH" ]
then
    stale_branch_error
    exit 1
fi
