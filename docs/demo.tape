# Records docs/demo.gif, which the README opens with.
#
#     vhs docs/demo.tape
#
# What it shows is the property the engine exists for: four clients keep
# producing tokens while a prompt of eight hundred words goes through, because
# that prompt is fed a slice per pass rather than taking a pass to itself. Run
# `python3 scripts/demo.py --chunk off` to watch the same thing stall.

Output docs/demo.gif

Set FontSize 15
Set Width 1020
Set Height 300
Set Padding 22
Set Framerate 12
Set TypingSpeed 40ms

Type "make demo"
Sleep 700ms
Enter
Sleep 9s
