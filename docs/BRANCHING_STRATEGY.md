# Branching Strategy

To maintain a clean and stable codebase, we follow a feature-branch workflow.

## Main Branches
- **`main`**: The stable production-ready branch. All code in this branch should be tested and deployable.

## Supporting Branches
- **Feature Branches** (`feature/name-of-feature`): Used for developing new features. Created off `main`.
- **Bug Fix Branches** (`fix/issue-description`): Used for fixing bugs. Created off `main`.
- **Hotfix Branches** (`hotfix/critical-issue`): Used for urgent fixes in production.

## Workflow
1. **Create a Branch**: 
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. **Commit Changes**: Use descriptive commit messages.
   ```bash
   git add .
   git commit -m "feat: add user authentication"
   ```
3. **Push to Remote**:
   ```bash
   git push -u origin feature/my-new-feature
   ```
4. **Open a Pull Request (PR)**:
   - Target the `main` branch.
   - Fill out the PR template.
   - Request review from a teammate.
5. **Merge**: Once approved and CI passes, merge into `main`.

## Commit Conventional
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests
