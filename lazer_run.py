#colorama
import asyncio
import colorama
from lazer.lazer import lazer

colorama.init()
print(colorama.ansi.clear_screen())
myLazer = lazer()
asyncio.run(myLazer.init_main(myLazer))