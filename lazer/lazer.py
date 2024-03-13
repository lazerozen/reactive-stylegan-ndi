from __future__ import print_function

import sys
sys.path.append("../stylegan3")
sys.path.append("../lazer")
from typing import List, Optional, Tuple, Union
import time
import dnnlib
import numpy as np
import torch
import re
import cv2 as cv
import numpy as np
import torch
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
    """
    The `lazer` class represents an object that generates and sends video frames using NDI (Network Device Interface).
    It utilizes the StyleGAN model to render images based on specified parameters and sends them as video frames via NDI.

    Attributes:
        osc_ip (str): The IP address for OSC (Open Sound Control) communication.
        osc_port (int): The port number for OSC communication.
        ndi_name (str): The name of the NDI source.
        latent (list): The x and y values of the latent vector used for generating images.
        mustTransform (bool): Flag indicating if a new frame needs to be rendered due to a transformation.
        translate_x (int): The translation value along the x-axis for the input transformation.
        translate_y (int): The translation value along the y-axis for the input transformation.
        rotation (int): The rotation angle for the input transformation.
        step_y (int): The step size for the latent walk.
        fps (int): The desired frames per second for rendering and sending video frames.
        mustRun (bool): Flag indicating if a new frame needs to be rendered and sent.
        printFrequency (int): The frequency in milliseconds for printing status information.
        renderArgs (dnnlib.EasyDict): The render arguments for the StyleGAN model.
        timeServerStart (float): The start time of the server.
        ndiFrequency (int): The number of NDI frames sent per second.
        ndiFramesSent (int): The total number of NDI frames sent.
        imagesGenerated (int): The total number of images generated.
        skippedLoop (int): The number of loops where no rendering was necessary.
        modelChanged (bool): Flag indicating if the StyleGAN model has changed and needs to be reloaded.
        randFactor (list): The random latent addition to add a random value to existing latents.
    """
class lazer(filterhandler):

    def __init__(self):
        super().__init__()

        # Initialize attributes
        self.osc_ip = "127.0.0.1"
        self.osc_port = 161
        self.ndi_name = 'stylegan3-lazer'
        self.latent = [random.randint(0,1611312),random.randint(0,1611312)]
        self.mustTransform = True
        self.translate_x = 0
        self.translate_y = 0
        self.rotation = 0
        self.step_y = 100
        self.fps = 15
        self.mustRun = True
        self.printFrequency = 50
        self.renderArgs = dnnlib.EasyDict(
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
        self.timeServerStart = time.perf_counter()
        self.ndiFrequency = 0
        self.ndiFramesSent = 0
        self.imagesGenerated = 0
        self.skippedLoop = 0
        self.modelChanged = True
        self.randFactor = [0,0]

    def sendVideo(self, img, ndi_send, video_frame):
        """
        Sends the video frame via NDI.

        Args:
            img (numpy.ndarray): The image data.
            ndi_send (ndi.SendInstance): The NDI send instance.
            video_frame (ndi.VideoFrameV2): The NDI video frame.

        Returns:
            int: 0 if successful.
        """
        video_frame.data = img
        video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBX
        ndi.send_send_video_v2(ndi_send, video_frame) 
        return 0

    def _get_pinned_buf(self, ref):
        """
        Helper function to get a pinned buffer.

        Args:
            ref (torch.Tensor): The reference tensor.

        Returns:
            torch.Tensor: The pinned buffer.
        """
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def printStatus(self):
        """
        Prints the current status.
        """
        print("Generating at "+str(1/(time.time()-self.lastRun))+" fps")
        print("Images generated: "+str(self.imagesGenerated))
        print("NDI frames sent: "+str(self.ndiFramesSent))
        print("Skipped loops: "+str(self.skippedLoop))
        print("LatentX: "+str(self.latent[0]))
        print("LatentY: "+str(self.latent[1]), end='')
        print("\033[A\033[A\033[A\033[A\033[A\033[A")

    def randomizeSeeds(self):
        """
        Randomizes the seeds for latent vector.
        """
        self.randFactor[0] = random.randint(0,1611312)
        self.randFactor[1] = random.randint(0,1611312)
        self.randomize = False
        return

    def setSeedsFromLatent(self):
        """
        Sets the seeds for latent vector based on the current latent coordinates.
        """
        randomizedLatentX = self.latent[0] + self.randFactor[0]
        randomizedLatentY = self.latent[1] + self.randFactor[1]

        self.renderArgs["w0_seeds"] = [] # [[seed, weight], ...]
        for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
            seed_x = np.floor(randomizedLatentX) + ofs_x
            seed_y = np.floor(randomizedLatentY) + ofs_y
            seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
            weight = (1 - abs(randomizedLatentX - seed_x)) * (1 - abs(randomizedLatentY - seed_y))
            if weight > 0:
                self.renderArgs["w0_seeds"].append([seed, weight])

    def initNdi(self):
        """
        Initializes the NDI communication.

        Returns:
            tuple: The NDI send instance and video frame.
        """
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
        """
        The main loop that renders and sends frames.
        """
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

            if localMustRun:
                self.lastRun = time.time()

                self.setSeedsFromLatent()

                res = renderer.render(**self.renderArgs)
                self.imagesGenerated += 1

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
        """
        Sets up the OSC dispatcher.

        Returns:
            osc.dispatcher.Dispatcher: The OSC dispatcher.
        """
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
        """
        Initializes the main program.

        Args:
            timer (float): The timer value.

        Returns:
            None
        """
        dispatcher = self.setupDispatcher()
        
        server = AsyncIOOSCUDPServer((self.osc_ip, self.osc_port), dispatcher, asyncio.get_event_loop())
        transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

        await self.loop()  # Enter main loop of program

        transport.close()  # Clean up serve endpoint

