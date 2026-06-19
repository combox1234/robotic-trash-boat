# How to Contribute using Fork and Pull Request

This guide explains the standard GitHub workflow for contributing to this project. If you are new to Git and GitHub, just follow these steps carefully!

## 1. Fork the Repository
A "Fork" is your own personal copy of the original project. 
1. Go to the main project page on GitHub (e.g., `https://github.com/ComBox360/robotic-trash-boat`).
2. In the top-right corner of the page, click the **Fork** button.
3. This creates a copy of the repository in your personal GitHub account (e.g., `https://github.com/your-username/robotic-trash-boat`).

## 2. Clone Your Fork Locally
Now you need to download your fork to your computer.
1. Go to your fork on GitHub.
2. Click the green **Code** button and copy the HTTPS URL.
3. Open your terminal or command prompt and run:
   ```bash
   git clone https://github.com/your-username/robotic-trash-boat.git
   ```
4. Navigate into the new folder:
   ```bash
   cd robotic-trash-boat
   ```

## 3. Create a Branch for Your Changes
Always create a new branch for your specific feature or fix. **Do not make changes directly to the `main` branch!**
```bash
git checkout -b my-new-feature
```
*(Replace `my-new-feature` with a short, descriptive name for what you are doing).*

## 4. Make Your Changes
Open the project in your code editor (like VS Code), write your code, test it, and save the files.

## 5. Commit Your Changes
Once you are happy with your edits, you need to save them in Git.
1. Add the changed files to the "staging area":
   ```bash
   git add .
   ```
   *(The `.` means "add all changed files". If you only want to add specific files, use `git add filename.py` instead).*
2. Create a "commit" with a message explaining what you did:
   ```bash
   git commit -m "Add a brief description of what you changed"
   ```

## 6. Push Your Changes to Your Fork
Upload your committed changes from your computer back to your fork on GitHub.
```bash
git push -u origin my-new-feature
```

## 7. Open a Pull Request (PR)
A Pull Request is how you ask the original project owners to review and include your changes.
1. Go to the original project repository on GitHub (the one you clicked "Fork" on).
2. You will see a green banner that says **Compare & pull request**. Click it.
3. Give your Pull Request a clear title and write a short description of what you added or fixed.
4. Click **Create pull request**.

## 8. Wait for Review
The project maintainers will review your code. They might ask for changes. If they do, simply make the edits on your computer, run `git add .`, `git commit`, and `git push` again. The Pull Request will update automatically!
