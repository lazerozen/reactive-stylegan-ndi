from __future__ import print_function

import sys
sys.path.append("../stylegan3")
sys.path.append("../lazer")
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
from stylegan3 import gen_images
import time
import NDIlib as ndi

#lazer class portions
from lazer._filterhandlers import filterhandler

class lazer(filterhandler):
    #osc config
    osc_ip = "127.0.0.1"
    osc_port = 161
    
    #ndi config
    ndi_name = 'stylegan3-lazer'

    #where we look into the model
    latent = [random.randint(0,1611312),random.randint(0,1611312)] #x,y

    #decides if a new frame has to rendered
    mustTransform = True
    translate_x = 0
    translate_y = 0
    rotation = 0

    #step size for latent walk
    step_y = 100

    #default fps
    fps = 15
    #decides if a new frame has to rendered
    mustRun = True

    #print frequency in ms
    printFrequency = 50

    #all render args for all the beauty
    renderArgs = dnnlib.EasyDict(
        pkl             = 'C:\\git\\stylegan3\\models\\wikiart-1024-stylegan3-t-17.2Mimg.pkl',
        w0_seeds        = [],
        trunc_psi       = 1.6,
        trunc_cutoff    = 16,
        random_seed     = 0,
        noise_mode      = 'const',
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_normalize   = False,
        input_transform = [[1,0,0],[0,1,0],[0,0,1]],
        untransform     = False,
        img_scale_db    = 0
        ) 
    
    timeServerStart = time.perf_counter()
    #number of ndi frames sent per second
    ndiFrequency = 0
    #number of ndi frames sent
    ndiFramesSent = 0
    #number of images generate
    imagesGenerated = 0
    #number of loops where no rendering was neccessary
    skippedLoop = 0
    # must redo model load stuff
    modelChanged = True
    # random latent addition to add a random value to existing latents
    randFactor = [0,0]

    def sendVideo (self,img,ndi_send,video_frame):
        video_frame.data = img
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

    def printStatus(self):
        print("Generating at "+str(1/(time.time()-self.lastRun))+" fps")
        print("Images generated: "+str(self.imagesGenerated))
        print("NDI frames sent: "+str(self.ndiFramesSent))
        print("skipped loops: "+str(self.skippedLoop))
        print("latentX: "+str(self.latent[0]))
        print("latentY: "+str(self.latent[1]),end='')
        
        print("\033[A\033[A\033[A\033[A\033[A\033[A")

    def randomizeSeeds(self):
        self.randFactor[0] = random.randint(0,1611312)
        self.randFactor[1] = random.randint(0,1611312)
        self.randomize = False
        return

    def setSeedsFromLatent(self):
        randomizedLatentX = self.latent[0]+self.randFactor[0]
        randomizedLatentY = self.latent[1]+self.randFactor[1]

        self.renderArgs["w0_seeds"] = [] # [[seed, weight], ...]
        for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
            seed_x = np.floor(randomizedLatentX) + ofs_x
            seed_y = np.floor(randomizedLatentY) + ofs_y
            seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
            weight = (1 - abs(randomizedLatentX - seed_x)) * (1 - abs(randomizedLatentY - seed_y))
            if weight > 0:
                self.renderArgs["w0_seeds"].append([seed, weight])
    
    def initNdi(self):
        if not ndi.initialize():
            print("Could not initialize NDI.")
            exit()
        
        send_settings = ndi.SendCreate()        
        send_settings.ndi_name = self.ndi_name
        send_settings.clock_video = False
        send_settings.clock_audio = False

        ndi_send = ndi.send_create(send_settings)
        
        if ndi_send is None:
            print("Could not create NDI send.")
            return 0
        
        video_frame = ndi.VideoFrameV2()

        return ndi_send, video_frame

    async def loop(self):
        ndi_send, video_frame = self.initNdi()

        renderer = Renderer()
        
        # to know when to flush the terminal after styegan init
        firstRun = True
        self.lastRun = time.time()
        lastPrint = self.lastRun

        while True:
            currentTime = time.time()

            # ms elapsed since last run
            elapsed = (currentTime - self.lastRun)*1000
            
            # run frequency goal in ms
            runEvery = 1000 / self.fps
            if (elapsed < runEvery):
                await asyncio.sleep(.001)
                continue

            # copy early for pseudo thread safety, consider lock
            localMustRun = self.mustRun
            self.mustRun = False

            if not firstRun:
                try:
                    # ms elapsed since last print
                    elapsedSincePrint = (currentTime - lastPrint)*1000
                    if elapsedSincePrint > self.printFrequency:
                        self.printStatus()
                        lastPrint = currentTime
                except:
                    x= 1
                finally:
                    x = 1
            if self.mustTransform:
                localMustRun = True
                transform = gen_images.make_transform([self.translate_x,self.translate_y],self.rotation)
                self.renderArgs["input_transform"] = transform
                self.mustTransform = False

            #if True:
            if localMustRun:
                self.lastRun = time.time()

                self.setSeedsFromLatent()

                res = renderer.render(**self.renderArgs)
                self.imagesGenerated+=1

                if firstRun:
                    #clear stylegan module output
                    print(colorama.ansi.clear_screen())
                    firstRun = False

                imgInColor = cv.cvtColor(res['image'], cv.COLOR_RGB2RGBA)
                
                #convert to 8bit if neccessary
                #ratio = np.amax(imgInColor) / 256 
                #imgInColor = (imgInColor / ratio).astype('uint8')

                if (self.modelChanged):
                    imgWidth, imgHeight, imgChannel = imgInColor.shape
                    video_frame.xres = imgWidth
                    video_frame.yres = imgHeight
                    self.modelChanged = False
                    #get rid of stylegan model load text
                    print(colorama.ansi.clear_screen())

                self.sendVideo(imgInColor,ndi_send, video_frame)
                self.ndiFramesSent += 1
            else:
                self.skippedLoop += 1

            await asyncio.sleep(.001)
    
    def setupDispatcher(self):
        dispatcher = Dispatcher()
        dispatcher.map("/latentX", self.filter_handler_latent_x)
        dispatcher.map("/latentY", self.filter_handler_latent_y)
        dispatcher.map("/truncpsi", self.filter_handler_truncpsi)
        dispatcher.map("/trunccutoff", self.filter_handler_trunccutoff)
        dispatcher.map("/inputtransformx", self.filter_handler_transformx)
        dispatcher.map("/inputtransformy", self.filter_handler_transformy)
        dispatcher.map("/inputtransformrotation", self.filter_handler_rotation)
        dispatcher.map("/imagescaledb", self.filter_handler_img_scale_db)
        dispatcher.map("/randomize", self.filter_handler_randomize)
        dispatcher.map("/targetfps", self.filter_handler_targetfps)
        dispatcher.map("/setpkl", self.filter_handler_setpkl)
        return dispatcher

    async def init_main(self, timer):
        dispatcher = self.setupDispatcher()
        
        server = AsyncIOOSCUDPServer((self.osc_ip, self.osc_port), dispatcher, asyncio.get_event_loop())
        transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

        await self.loop()  # Enter main loop of program

        transport.close()  # Clean up serve endpoint

