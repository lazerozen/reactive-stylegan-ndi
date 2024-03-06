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
#import dnnlib
from stylegan3.torch_utils import misc
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import asyncio
import random

#import fixpath
import colorama
from colorama import Fore, Back, Style, Cursor
from random import randint, choice
from string import printable

from stylegan3.viz.renderer import Renderer
from stylegan3 import dnnlib


import NDIlib as ndi

class lazer:
    latent = 0.1
    randomize = False
    fps = 15
    mustRun = True,

    renderArgs = dnnlib.EasyDict(
        #pkl='C:\\Users\\wolfg\\.cache\\dnnlib\\downloads\\ee187625eea60ab45ed60218c4f6874b_https___api.ngc.nvidia.com_v2_models_nvidia_research_stylegan3_versions_1_files_stylegan3-t-ffhq-1024x1024.pkl',
        #pkl = "C:\\Users\\wolfg\\.cache\\dnnlib\\downloads\\fc4ecbe86efeec3ce080e1ef7f7e0c20_https___api.ngc.nvidia.com_v2_models_nvidia_research_stylegan2_versions_1_files_stylegan2-afhqdog-512x512.pkl",
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

    def sendVideo (self,img,ndi_send,video_frame):
        video_frame.data = img
        #video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
        video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBX

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

    def printStatus(self, fps):
        print("Generating at "+fps+" fps")
        print("latent: "+str(self.latent))
        print("randomize: "+str(self.randomize),end='')
        
        print("\033[A\033[A\033[A")

    async def loop(self):
        #await loop()  # Enter main loop of program
        #asyncio.run(loop())  # Enter main loop of program
        if not ndi.initialize():
            print("Could not initialize NDI.")
            return 0

        send_settings = ndi.SendCreate()        
        send_settings.ndi_name = 'stylegan3-lazer'

        ndi_send = ndi.send_create(send_settings)
        
        if ndi_send is None:
            print("Could not create NDI send.")
            return 0
        
        video_frame = ndi.VideoFrameV2()

        renderer = Renderer()
        previousLatent = self.latent
        mustRun = True
        lastRun = time.time()
        lastGen = lastRun
        # to know when to flush the terminal after styegan init
        firstRun = True

        while True:
            # ms elapsed since last run
            elapsed = (time.time() - lastRun)*1000
            
            # run frequency goal in ms
            runEvery = 1000 / self.fps
            if (elapsed < runEvery):
                await asyncio.sleep(.001)
                continue

            # copy early for pseudo thread safety, consider lock
            localMustRun = self.mustRun
            if self.latent != previousLatent:
                localMustRun = True
            self.mustRun = False

            #self.move(1,40)
            #print('loop start',end='')
            if self.randomize:
                #self.renderArgs['random_seed'] = random.randint(0,1611312)
                myLazer.renderArgs['w0_seeds'][0][0] = random.randint(0,1611312)
                myLazer.renderArgs['w0_seeds'][1][0] = random.randint(0,1611312)
                myLazer.renderArgs['w0_seeds'][2][0] = random.randint(0,1611312)
                myLazer.renderArgs['w0_seeds'][3][0] = random.randint(0,1611312)
                self.randomize = False
                localMustRun = True

            c = None                                # class labels (not used in this example)
            
            if localMustRun:
                startGen = time.time()
                #self.move(1,1)
                if not firstRun:
                    self.printStatus(str(1/(time.time()-lastGen)))

                res = renderer.render(**self.renderArgs)
                
                if firstRun:
                    print(colorama.ansi.clear_screen())
                    firstRun = False

                #self.move(2,0)
                #print('Generation took %1.2f ms' % (time.time() - startGen)* 1000,end='')
                imgInColor = cv.cvtColor(res['image'], cv.COLOR_RGB2RGBA)
                self.sendVideo(imgInColor,ndi_send, video_frame)

                lastGen = time.time()

            #self.move(1,40)
            #print('loop done',end='')
            await asyncio.sleep(.001)

            previousLatent = self.latent
            lastRun = time.time()

    def filter_handler_latent_x(address, *args):
        # We expect one float argument
        if not len(args) == 1 or type(args[0]) is not float:
            return
        myLazer.renderArgs['w0_seeds'][0][1] = args[0]
        myLazer.renderArgs['w0_seeds'][1][1] = args[0]
        myLazer.mustRun = True
        #print('latent set to '+str(myLazer.latent), end='\r')
    
    def filter_handler_latent_y(address, *args):
        # We expect one float argument
        if not len(args) == 1 or type(args[0]) is not float:
            return
        myLazer.renderArgs['w0_seeds'][2][1] = args[0]
        myLazer.renderArgs['w0_seeds'][3][1] = args[0]
        myLazer.mustRun = True
        #print('latent set to '+str(myLazer.latent), end='\r')
    

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