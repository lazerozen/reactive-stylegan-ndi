import random
import os
from pythonosc.udp_client import SimpleUDPClient

class filterhandler:
	def filter_handler_latent_x(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		if self.latent[0] == args[0]:
			return
		
		self.latent[0] = args[0]
		self.mustRun = True
		return

	def filter_handler_latent_y(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		if self.latent[1] == args[0]:
			return
		
		self.latent[1] = args[0]
		self.mustRun = True
		return

	#-1 - 2 default 1
	def filter_handler_truncpsi(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['trunc_psi'] == args[0]:
			return
		
		self.renderArgs['trunc_psi'] = args[0]
		self.mustRun = True


	#input transform x
	def filter_handler_transformx(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.translate_x == args[0]:
			return
		
		self.translate_x = args[0]
		self.mustTransform = True

	#input transform y
	def filter_handler_transformy(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.translate_y == args[0]:
			return
		
		self.translate_y = args[0]
		self.mustTransform = True
		
	#input transform rotation
	def filter_handler_rotation(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.rotation == args[0]:
			return
		
		self.rotation = args[0]
		self.mustTransform = True

	# 0- 16 default 16
	def filter_handler_trunccutoff(self, address, *args):
		# We expect one int argument
		if not len(args) == 1 or type(args[0]) is not int:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['trunc_cutoff'] == args[0]:
			return
		
		self.renderArgs['trunc_cutoff'] = args[0]
		self.mustRun = True

	# -40 - 40 default 0
	def filter_handler_img_scale_db(self, address, *args):
		# We expect one int argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['img_scale_db'] == args[0]:
			return
		
		self.renderArgs['img_scale_db'] = args[0]
		self.mustRun = True

	def filter_handler_randomize(self, address, *args):
		self.randFactor[0] = random.randint(0,1611312)
		self.randFactor[1] = random.randint(0,1611312)
		self.mustRun = True

	def filter_handler_targetfps(self, address, *args):
		if not len(args) == 1 or type(args[0]) is not int:
			return
		self.fps = args[0]

	def filter_handler_setpkl(self, address, *args):
		if not len(args) == 1 or type(args[0]) is not str:
			return
		
		if (not os.path.isfile(args[0])):
			print ("Could not load pkl file "+args[0]+" - file does not exist")
			return

		print('loading pklfile '+args[0], end='\r')
		self.renderArgs.pkl = args[0]
		self.modelChanged = True
		self.mustRun = True
	
	# sends current latents including random factor via OSC to port 162
	def filter_handler_get_favorite_data (self, address, *args):
		ip = "127.0.0.1"
		port = 162

		client = SimpleUDPClient(ip, port)  # Create client

		paramsDict = {"latentX":self.latent[0]+self.randFactor[0]+ self.latentOffset[0], "latentY":self.latent[1]+self.randFactor[1]+ self.latentOffset[1]}
		paramsString = str(paramsDict)
		#client.send_message("/favorite_data", str(jsonPackage))   # Send float message
		client.send_message("/favorite_data", paramsString)

	def filter_handler_set_latent_offset (self, address, *args):
		if not len(args) == 2:
			return
		
		try:
			favoriteX = float(args[0])
			favoriteY = float(args[1])
		except ValueError:
			return

		
		#TODO: fail gracefully before cast
		self.latentOffset[0] = favoriteX - self.latent[0] - self.randFactor[0]
		self.latentOffset[1] = favoriteY - self.latent[1] - self.randFactor[1]
		self.mustRun = True