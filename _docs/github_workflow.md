# GitHub workflow — the pull-request loop

Every task in this project follows the same loop: make a **branch**, do the work,
**commit**, **push**, open a **pull request (PR)**, get it reviewed, **merge**. We never
commit directly to `main`. This doc is your cheat sheet.

## Why branches + PRs?

`main` is the "good, agreed-on" version of the project. You do your work on a *branch* (a
parallel copy), and a PR is how you propose folding your branch back into `main` — with a
chance for Taylor to review it first. It's how teams avoid stepping on each other and how
mistakes get caught before they land.

## One-time setup

```bash
git clone <repo-url>          # download the repo (Taylor gives you the URL)
cd multitenant-inspection-prediction
```

## The loop, every task

1. **Start from an up-to-date `main`:**
   ```bash
   git switch main
   git pull
   ```
2. **Create a branch** named for the task:
   ```bash
   git switch -c eda-bottom10
   ```
   (Short, descriptive, hyphenated. One branch per task / per PR.)
3. **Do the work** — edit the notebook or docs.
4. **See what changed / stage / commit:**
   ```bash
   git status                       # what changed
   git add _code/01_replicate_bottom10.ipynb
   git commit -m "Add EDA: class balance and feature distributions"
   ```
   Commit in small, logical chunks with clear messages. Double-check `git status` shows
   **no data files** (`.csv`, `.xlsx`) — `.gitignore` should prevent it, but verify.
5. **Push the branch:**
   ```bash
   git push -u origin eda-bottom10
   ```
6. **Open the PR.** Either click the link that `git push` prints, or on GitHub press the
   "Compare & pull request" button. Give it a short title and a sentence or two on what
   you did and anything you're unsure about. Request **Taylor** (twcrockett) as a reviewer.
7. **Respond to review.** Taylor may leave comments. To address them, just commit more
   changes on the *same branch* and push again — the PR updates automatically:
   ```bash
   git add -A && git commit -m "Address review: stratify the split" && git push
   ```
8. **Merge.** Once approved, click **Merge** on GitHub. Then start your next task back at
   step 1.

## Handy commands

| I want to… | Command |
|---|---|
| See what changed | `git status` / `git diff` |
| See recent commits | `git log --oneline` |
| Switch branches | `git switch <branch>` |
| Throw away changes to a file | `git restore <file>` |
| Update my branch with the latest `main` | `git switch main && git pull && git switch <branch> && git merge main` |

## Rules of thumb

- **Small PRs.** Easier to review, faster to merge. The "first tasks" list in the README
  is sized so each item is roughly one PR.
- **Never commit data.** If `git status` ever lists a `.csv`/`.xlsx`, stop — don't add it.
- **Pull before you branch.** Always start a new task from a fresh `main`.
- **Ask early.** A draft PR with a question is better than a day stuck.
