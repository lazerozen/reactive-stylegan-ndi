from __future__ import print_function
import os
import re
import sys
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
import dnnlib
from torch_utils import misc
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import asyncio


#import fixpath
import colorama
from colorama import Fore, Back, Style, Cursor
from random import randint, choice
from string import printable


import NDIlib as ndi

class lazer:
    latent = 0.1
    randomize = False

    def sendVideo (self,img,ndi_send,video_frame):
        video_frame.data = img
        #video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
        video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBX

        #start = time.time()
        #while time.time() - start < 60 * 5:

        ndi.send_send_video_v2(ndi_send, video_frame)

        #start_send = time.time()

        #for idx in reversed(range(200)):
            #img.fill(255 if idx % 2 else 0)
            #   ndi.send_send_video_v2(ndi_send, video_frame)

        #print('200 frames sent, at %1.2ffps' % (200.0 / (time.time() - start_send)))


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

    async def loop(self):
        #await loop()  # Enter main loop of program
        #asyncio.run(loop())  # Enter main loop of program
        if not ndi.initialize():
            print("meh1")
            return 0

        ndi_send = ndi.send_create()

        if ndi_send is None:
            print("meh2")
            return 0
        video_frame = ndi.VideoFrameV2()

        with open('C:\\Users\\wolfg\\.cache\\dnnlib\\downloads\\ee187625eea60ab45ed60218c4f6874b_https___api.ngc.nvidia.com_v2_models_nvidia_research_stylegan3_versions_1_files_stylegan3-t-ffhq-1024x1024.pkl', 'rb') as f:
            G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
            z = torch.randn([1, G.z_dim]).cuda()    # latent codes
            
            while True:
                self.move(1,40)
                print('loop start',end='')
                if self.randomize:
                    z = torch.randn([1, G.z_dim]).cuda()    # latent codes
                    self.randomize = False

                c = None                                # class labels (not used in this example)
                img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1], no truncation
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                imgInColor = cv.cvtColor(img[0].cpu().numpy(), cv.COLOR_RGB2RGBA)
                self.sendVideo(imgInColor,ndi_send, video_frame)

                for i in (1,100):
                    z.H[i] = self.latent
                self.move(1,40)
                print('loop done',end='')
                await asyncio.sleep(.02)

    def filter_handler_latent(address, *args):
        # We expect one float argument
        if not len(args) == 1 or type(args[0]) is not float:
            return
        
        myLazer.latent = args[0]
        print('latent set to '+str(myLazer.latent), end='\r')
        #print(f"{address}: {args}")
    
    def filter_handler_randomize(address, *args):
        print('randomize activated', end='\r')
        myLazer.randomize = True
        #print(f"{address}: {args}")

    dispatcher = Dispatcher()
    dispatcher.map("/latent", filter_handler_latent)
    dispatcher.map("/randomize", filter_handler_randomize)

    ip = "127.0.0.1"
    port = 1337

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