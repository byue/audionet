## Branching Strategy
This repo is organized into 3 permanent branches: dev, master, and deploy.  All other branches are considered to be feature branches.  Feature branches are created by branching from dev, and they're the only place where active development may take place.  Be sure to issue a git pull before creating a feature branch from dev.  Once a feature is complete, then issue a pull request to merge the feature branch back into dev.  On approval dev will be merged onto master and deploy.  See the diagram below.

![Feature](images/feature.png)

If production hotfixes are required outside of the normal development cycle then create a hotfix branch from master, implement the fix, and mege the hotfix branch back to master and onto deploy.  The hotfix commit will then need to be cherry picked onto dev.  See the diagram below.

![Hotfix](images/hotfix.png)

## Branches
* feature - branches that contain a single logical unit of work
* dev - branch used for aggregating features across developers
* master - represents the code that is deployed in production
* deploy - a hot branch that triggers a production deployment

## Githooks
Run the following command from the repository root to install the appropriate githooks.
```
bash ./tools/githooks/install.sh
```
