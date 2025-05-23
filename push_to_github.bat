@echo off
echo Initializing git repository...
git init
echo Adding files to git...
git add .
echo Configuring git user...
git config --global user.email "temp@example.com"
git config --global user.name "Temporary User"
echo Creating initial commit...
git commit -m "Initial commit"
echo Adding remote repository...
git remote add origin https://github.com/AbdellahRAISSOUNI/projet-AIML-gi2.git
echo Pushing to github...
git push -u origin master
echo Done.
pause 