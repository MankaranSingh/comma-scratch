import numpy as np 
import cv2
import pandas as pd

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(winSize  = (15, 15), 
		#maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.031))

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2


class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy, 
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
	def __init__(self, cam, meta_data_path):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.x, self.y, self.z = 0, 0, 0
		self.theta = 0
		#self.detector = cv2.ORB_create()
		#self.detector = cv2.xfeatures2d.SURF_create()
		self.detector = cv2.xfeatures2d.SIFT_create()
		self.annotations = pd.read_csv(meta_data_path)
		self.annotations['timestamp'] = self.annotations['timestamp']/1000000000

	def getAbsoluteScale(self, frame_id):  #specialized Udacity dataset
		v_prev = self.annotations['speed'].values[frame_id-1]
		t_prev = self.annotations['timestamp'].values[frame_id-1]
		t_curr = self.annotations['timestamp'].values[frame_id]
		dt = t_curr - t_prev

		return v_prev * dt * 2.4 + 0.3

	def processFirstFrame(self):
		self.px_ref = self.detector.detect(self.new_frame)
		self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.frame_stage = STAGE_DEFAULT_FRAME 
		self.px_ref = self.px_cur

	def processFrame(self, frame_id):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		absolute_scale = self.getAbsoluteScale(frame_id)
		theta = np.arctan2(t[2], t[0])
		if(absolute_scale > 0.1) :
			self.cur_t = self.cur_t + absolute_scale*self.cur_R.dot(t) 
			self.cur_R = R.dot(self.cur_R)
			self.theta = np.arctan2(self.cur_t[2], self.cur_t[0])
		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur

	def update(self, img, frame_id):
		self.new_frame = img
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(frame_id)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame
