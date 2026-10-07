# Terminal demo of `dc behav`. Regenerate from the repo root with:
#   vhs docs/behav.tape
Output docs/behav.gif

Set Shell "bash"
Set FontSize 14
Set Width 1000
Set Height 440
Set Padding 16
Set TypingSpeed 40ms
Set Theme "Catppuccin Mocha"

Hide
Type "export PATH=$PWD/.venv/bin:$PATH VIRTUAL_ENV= PS1='$ ' && cd $(mktemp -d) && clear"
Enter
Show

Type "dc behav probe ~/data/test/facemap/raw"
Sleep 300ms
Enter
Sleep 3s

Type "clear"
Enter
Type "dc behav compress ~/data/test/facemap/raw run000.mp4"
Sleep 300ms
Enter
Sleep 9s

Type "du -sh ~/data/test/facemap/raw run000.mp4"
Sleep 300ms
Enter
Sleep 3s
