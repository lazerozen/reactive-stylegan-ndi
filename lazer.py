from __future__ import print_function

import sys
sys.path.append("stylegan3")
import os
import re
from typing import List, Optional, Tuple, Union
import time
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import click
import pickle
import re
import cv2 as cv
import copy
import numpy as np
import torch
from threading import Thread
from stylegan3.torch_utils import misc
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import asyncio
import random
import colorama
from colorama import Fore, Back, Style, Cursor
from random import randint, choice
from string import printable
from stylegan3.viz.renderer import Renderer
from stylegan3 import dnnlib
import time

import NDIlib as ndi

class lazer:
    latent = 0.1
    randomize = False
    fps = 15
    mustRun = True,

    renderArgs = dnnlib.EasyDict(
        pkl = 'C:\\git\\stylegan3\\models\\wikiart-1024-stylegan3-t-17.2Mimg.pkl',
        w0_seeds=[[213, 0.23870113378660512], [214, 0.7403464852610139], [313, 0.005108390022670563], [314, 0.01584399092971049]],
        trunc_psi=1.6,
        trunc_cutoff = 16,
        random_seed = 0,
        noise_mode = 'const',
        force_fp32 = False,
        layer_name = None,
        sel_channels = 3,
        base_channel = 0,
        img_scale_db = 0,
        img_normalize = False,
        #input_transform,
        untransform = False
        ) 
    
    timeServerStart = time.perf_counter()
    ndiFrequency = 0
    ndiFramesSent = 0
    imagesGenerated = 0
    skippedLoop = 0

    def sendVideo (self,img,ndi_send,video_frame):
        video_frame.data = img
        video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA
        ndi.send_send_video_v2(ndi_send, video_frame) 
        return 0

    #TODO!
    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf
    
    # cursor movement
    def move (self, y, x):
        print("\033[%d;%dH" % (y, x))

    def printStatus(self):
        print("Generating at "+str(1/(time.time()-self.lastRun))+" fps")
        print("Images generated: "+str(self.imagesGenerated))
        print("NDI frames sent: "+str(self.ndiFramesSent))
        print("skipped loops: "+str(self.skippedLoop))
        print("latentX: "+str(self.renderArgs['w0_seeds'][0][1]))
        print("latentY: "+str(self.renderArgs['w0_seeds'][2][1]))
        print("randomize: "+str(self.randomize),end='')
        
        print("\033[A\033[A\033[A\033[A\033[A\033[A\033[A")

    def randomizeSeeds(self):
        myLazer.renderArgs['w0_seeds'][0][0] = random.randint(0,1611312)
        myLazer.renderArgs['w0_seeds'][1][0] = random.randint(0,1611312)
        myLazer.renderArgs['w0_seeds'][2][0] = random.randint(0,1611312)
        myLazer.renderArgs['w0_seeds'][3][0] = random.randint(0,1611312)
        self.randomize = False

    async def loop(self):
        if not ndi.initialize():
            print("Could not initialize NDI.")
            return 0

        send_settings = ndi.SendCreate()        
        send_settings.ndi_name = 'stylegan3-lazer'
        send_settings.clock_video = False
        send_settings.clock_audio = False

        ndi_send = ndi.send_create(send_settings)
        
        if ndi_send is None:
            print("Could not create NDI send.")
            return 0
        
        video_frame = ndi.VideoFrameV2()

        renderer = Renderer()
        
        # to know when to flush the terminal after styegan init
        firstRun = True
        self.lastRun = time.time()
            
        while True:
            # ms elapsed since last run
            elapsed = (time.time() - self.lastRun)*1000
            
            # run frequency goal in ms
            runEvery = 1000 / self.fps
            if (elapsed < runEvery):
                await asyncio.sleep(.001)
                continue

            # copy early for pseudo thread safety, consider lock
            localMustRun = self.mustRun
            self.mustRun = False

            if self.randomize:
                self.randomizeSeeds()
                localMustRun = True

            if not firstRun:
                try:
                    #if faster than min resolution of timer
                    self.printStatus()
                except:
                    x= 1
                finally:
                    x = 1

            #if True:
            if localMustRun:
                self.lastRun = time.time()
                res = renderer.render(**self.renderArgs)
                self.imagesGenerated+=1

                if firstRun:
                    #clear stylegan module output
                    print(colorama.ansi.clear_screen())
                    firstRun = False

                imgInColor = cv.cvtColor(res['image'], cv.COLOR_RGB2RGBA)
                # TODO only after pkl load
                imgWidth, imgHeight, imgChannel = imgInColor.shape
                video_frame.xres = imgWidth
                video_frame.yres = imgHeight
                self.sendVideo(imgInColor,ndi_send, video_frame)
                
                self.ndiFramesSent += 1
            else:
                self.skippedLoop += 1

            await asyncio.sleep(.001)

    def filter_handler_latent_x(address, *args):
        # We expect one float argument
        if not len(args) == 1 or type(args[0]) is not float:
            return
        
        # do nothing if latens are the same
        if myLazer.renderArgs['w0_seeds'][0][1] == args[0] and myLazer.renderArgs['w0_seeds'][1][1] == args[0] :
            return
        
        myLazer.renderArgs['w0_seeds'][0][1] = args[0]
        myLazer.renderArgs['w0_seeds'][1][1] = args[0]
        myLazer.mustRun = True
    
    def filter_handler_latent_y(address, *args):
        # We expect one float argument
        if not len(args) == 1 or type(args[0]) is not float:
            return
        
        # do nothing if latens are the same
        if myLazer.renderArgs['w0_seeds'][2][1] == args[0] and myLazer.renderArgs['w0_seeds'][3][1] == args[0] :
            return
        
        myLazer.renderArgs['w0_seeds'][2][1] = args[0]
        myLazer.renderArgs['w0_seeds'][3][1] = args[0]
        myLazer.mustRun = True

    def filter_handler_randomize(address, *args):
        print('randomize activated', end='\r')
        myLazer.randomize = True

    def filter_handler_targetfps(address, *args):
        if not len(args) == 1 or type(args[0]) is not int:
            return
        print('set fps to '+str(args[0]), end='\r')
        myLazer.fps = args[0]

    def filter_handler_setpkl(address, *args):
        if not len(args) == 1 or type(args[0]) is not str:
            return
        
        if (not os.path.isfile(args[0])):
            print ("Could not load pkl file "+args[0]+" - file does not exist")

        print('loading pklfile '+args[0], end='\r')
        myLazer.renderArgs.pkl = args[0]

    dispatcher = Dispatcher()
    dispatcher.map("/latentX", filter_handler_latent_x)
    dispatcher.map("/latentY", filter_handler_latent_y)
    dispatcher.map("/randomize", filter_handler_randomize)
    dispatcher.map("/targetfps", filter_handler_targetfps)
    dispatcher.map("/setpkl", filter_handler_setpkl)

    ip = "127.0.0.1"
    port = 161

    async def init_main(self, timer):
        server = AsyncIOOSCUDPServer((self.ip, self.port), self.dispatcher, asyncio.get_event_loop())
        transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

        await self.loop()  # Enter main loop of program

        transport.close()  # Clean up serve endpoint

#colorama
colorama.init()
print(colorama.ansi.clear_screen())
myLazer = lazer()
asyncio.run(myLazer.init_main(myLazer))