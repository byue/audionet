#!/bin/bash
set -e

cp ./tools/githooks/common.sh ./.git/hooks/common.sh
cp ./tools/githooks/post-checkout.sh ./.git/hooks/post-checkout
cp ./tools/githooks/pre-commit.sh ./.git/hooks/pre-commit

echo "Githooks install complete.\n"
